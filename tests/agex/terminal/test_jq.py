"""Tests for jq command."""

import pytest

from agex.fs.virtual import VirtualFS
from agex.state import Live
from agex.terminal.interpreter import execute_script
from agex.terminal.interpreter.datatypes import TerminalError
from agex.terminal.jq import ParseError, parse_filter
from agex.terminal.parser import to_script


@pytest.fixture
def fs():
    state = Live()
    return VirtualFS(state)


# =============================================================================
# Parser tests
# =============================================================================


class TestParser:
    def test_identity(self):
        expr = parse_filter(".")
        assert expr.__class__.__name__ == "Identity"

    def test_field(self):
        expr = parse_filter(".name")
        assert expr.__class__.__name__ == "Field"
        assert expr.name == "name"

    def test_nested_field(self):
        expr = parse_filter(".foo.bar")
        assert expr.__class__.__name__ == "Pipe"

    def test_index(self):
        expr = parse_filter(".[0]")
        assert expr.__class__.__name__ == "Index"
        assert expr.index == 0

    def test_negative_index(self):
        expr = parse_filter(".[-1]")
        assert expr.__class__.__name__ == "Index"
        assert expr.index == -1

    def test_iterate(self):
        expr = parse_filter(".[]")
        assert expr.__class__.__name__ == "Iterate"

    def test_pipe(self):
        expr = parse_filter(".foo | .bar")
        assert expr.__class__.__name__ == "Pipe"

    def test_function_call(self):
        expr = parse_filter("keys")
        assert expr.__class__.__name__ == "FunctionCall"
        assert expr.name == "keys"

    def test_function_with_args(self):
        expr = parse_filter("select(.active)")
        assert expr.__class__.__name__ == "FunctionCall"
        assert expr.name == "select"
        assert len(expr.args) == 1

    def test_comparison(self):
        expr = parse_filter(".x == 1")
        assert expr.__class__.__name__ == "Comparison"

    def test_boolean_ops(self):
        expr = parse_filter(".a and .b")
        assert expr.__class__.__name__ == "BoolOp"

    def test_literal_string(self):
        expr = parse_filter('"hello"')
        assert expr.__class__.__name__ == "Literal"
        assert expr.value == "hello"

    def test_literal_number(self):
        expr = parse_filter("42")
        assert expr.__class__.__name__ == "Literal"
        assert expr.value == 42

    def test_array_construct(self):
        # Directly test parser, not via shell
        expr = parse_filter("[.foo]")
        assert expr.__class__.__name__ == "ArrayConstruct"

    def test_object_construct(self):
        expr = parse_filter("{name: .title}")
        assert expr.__class__.__name__ == "ObjectConstruct"

    def test_conditional(self):
        expr = parse_filter("if .x then .a else .b end")
        assert expr.__class__.__name__ == "Conditional"

    def test_parse_error(self):
        with pytest.raises(ParseError):
            parse_filter(".[")


# =============================================================================
# Basic command tests
# =============================================================================


class TestBasicCommands:
    def test_pretty_print(self, fs):
        fs.write("data.json", b'{"name": "test", "count": 42}')
        output = execute_script(to_script('jq "." data.json'), fs)
        assert '"name": "test"' in output
        assert '"count": 42' in output

    def test_compact_output(self, fs):
        fs.write("data.json", b'{"a": 1, "b": 2}')
        output = execute_script(to_script('jq -c "." data.json'), fs)
        assert output.strip() == '{"a":1,"b":2}'

    def test_field_access(self, fs):
        fs.write("data.json", b'{"name": "test"}')
        output = execute_script(to_script('jq ".name" data.json'), fs)
        assert output.strip() == '"test"'

    def test_raw_output(self, fs):
        fs.write("data.json", b'{"name": "test"}')
        output = execute_script(to_script('jq -r ".name" data.json'), fs)
        assert output.strip() == "test"

    def test_nested_field(self, fs):
        fs.write("data.json", b'{"config": {"host": "localhost"}}')
        output = execute_script(to_script('jq ".config.host" data.json'), fs)
        assert "localhost" in output

    def test_array_index(self, fs):
        fs.write("data.json", b'["a", "b", "c"]')
        output = execute_script(to_script('jq ".[1]" data.json'), fs)
        assert output.strip() == '"b"'

    def test_negative_index(self, fs):
        fs.write("data.json", b'["a", "b", "c"]')
        output = execute_script(to_script('jq ".[-1]" data.json'), fs)
        assert output.strip() == '"c"'

    def test_stdin_input(self, fs):
        fs.write("data.json", b'{"x": 42}')
        output = execute_script(to_script('cat data.json | jq ".x"'), fs)
        assert output.strip() == "42"

    def test_null_input(self, fs):
        output = execute_script(to_script('jq -n "1 + 2"'), fs)
        assert output.strip() == "3"


# =============================================================================
# Array/iteration tests
# =============================================================================


class TestArrayIteration:
    def test_iterate_array(self, fs):
        fs.write("data.json", b"[1, 2, 3]")
        output = execute_script(to_script('jq ".[]" data.json'), fs)
        assert "1" in output
        assert "2" in output
        assert "3" in output

    def test_iterate_and_access(self, fs):
        fs.write("data.json", b'[{"name": "a"}, {"name": "b"}]')
        output = execute_script(to_script('jq -r ".[] | .name" data.json'), fs)
        lines = output.strip().split("\n")
        assert lines == ["a", "b"]

    def test_slice(self, fs):
        fs.write("data.json", b"[1, 2, 3, 4, 5]")
        output = execute_script(to_script('jq ".[1:3]" data.json'), fs)
        assert "2" in output
        assert "3" in output
        assert "1" not in output.split("[")[1]  # 1 not in result

    def test_slice_from_start(self, fs):
        fs.write("data.json", b"[1, 2, 3, 4, 5]")
        output = execute_script(to_script('jq ".[:2]" data.json'), fs)
        assert "[" in output
        assert "1" in output
        assert "2" in output

    def test_slice_to_end(self, fs):
        fs.write("data.json", b"[1, 2, 3, 4, 5]")
        output = execute_script(to_script('jq ".[3:]" data.json'), fs)
        assert "4" in output
        assert "5" in output


# =============================================================================
# Function tests
# =============================================================================


class TestFunctions:
    def test_keys(self, fs):
        fs.write("data.json", b'{"z": 1, "a": 2}')
        output = execute_script(to_script('jq "keys" data.json'), fs)
        assert '"a"' in output
        assert '"z"' in output

    def test_length_array(self, fs):
        fs.write("data.json", b"[1, 2, 3]")
        output = execute_script(to_script('jq "length" data.json'), fs)
        assert output.strip() == "3"

    def test_length_object(self, fs):
        fs.write("data.json", b'{"a": 1, "b": 2}')
        output = execute_script(to_script('jq "length" data.json'), fs)
        assert output.strip() == "2"

    def test_length_string(self, fs):
        fs.write("data.json", b'"hello"')
        output = execute_script(to_script('jq "length" data.json'), fs)
        assert output.strip() == "5"

    def test_type(self, fs):
        fs.write("data.json", b'{"a": 1}')
        output = execute_script(to_script('jq "type" data.json'), fs)
        assert output.strip() == '"object"'

    def test_select(self, fs):
        fs.write("data.json", b'[{"x": 1}, {"x": 2}, {"x": 3}]')
        output = execute_script(to_script('jq ".[] | select(.x > 1)" data.json'), fs)
        assert '"x": 2' in output
        assert '"x": 3' in output
        assert '"x": 1' not in output

    def test_map(self, fs):
        fs.write("data.json", b"[1, 2, 3]")
        output = execute_script(to_script('jq "map(. * 2)" data.json'), fs)
        assert "2" in output
        assert "4" in output
        assert "6" in output

    def test_sort(self, fs):
        fs.write("data.json", b"[3, 1, 2]")
        output = execute_script(to_script('jq "sort" data.json'), fs)
        # Check order
        idx_1 = output.index("1")
        idx_2 = output.index("2")
        idx_3 = output.index("3")
        assert idx_1 < idx_2 < idx_3

    def test_unique(self, fs):
        fs.write("data.json", b"[1, 2, 1, 3, 2]")
        output = execute_script(to_script('jq "unique" data.json'), fs)
        assert output.count("1") == 1
        assert output.count("2") == 1

    def test_reverse(self, fs):
        fs.write("data.json", b"[1, 2, 3]")
        output = execute_script(to_script('jq "reverse" data.json'), fs)
        idx_1 = output.index("1")
        idx_3 = output.index("3")
        assert idx_3 < idx_1

    def test_first(self, fs):
        fs.write("data.json", b"[1, 2, 3]")
        output = execute_script(to_script('jq "first" data.json'), fs)
        assert output.strip() == "1"

    def test_last(self, fs):
        fs.write("data.json", b"[1, 2, 3]")
        output = execute_script(to_script('jq "last" data.json'), fs)
        assert output.strip() == "3"

    def test_add(self, fs):
        fs.write("data.json", b"[1, 2, 3]")
        output = execute_script(to_script('jq "add" data.json'), fs)
        assert output.strip() == "6"

    def test_min_max(self, fs):
        fs.write("data.json", b"[3, 1, 2]")
        output = execute_script(to_script('jq "min" data.json'), fs)
        assert output.strip() == "1"
        output = execute_script(to_script('jq "max" data.json'), fs)
        assert output.strip() == "3"

    def test_join(self, fs):
        fs.write("data.json", b'["a", "b", "c"]')
        output = execute_script(to_script("jq 'join(\"-\")' data.json"), fs)
        assert output.strip() == '"a-b-c"'

    def test_split(self, fs):
        fs.write("data.json", b'"a-b-c"')
        output = execute_script(to_script("jq 'split(\"-\")' data.json"), fs)
        assert '"a"' in output
        assert '"b"' in output
        assert '"c"' in output

    def test_to_entries(self, fs):
        fs.write("data.json", b'{"a": 1}')
        output = execute_script(to_script('jq "to_entries" data.json'), fs)
        assert '"key": "a"' in output
        assert '"value": 1' in output

    def test_from_entries(self, fs):
        fs.write("data.json", b'[{"key": "a", "value": 1}]')
        output = execute_script(to_script('jq "from_entries" data.json'), fs)
        assert '"a": 1' in output

    def test_has(self, fs):
        fs.write("data.json", b'{"a": 1}')
        output = execute_script(to_script("jq 'has(\"a\")' data.json"), fs)
        assert output.strip() == "true"
        output = execute_script(to_script("jq 'has(\"b\")' data.json"), fs)
        assert output.strip() == "false"


# =============================================================================
# Construction tests
# =============================================================================


class TestConstruction:
    def test_object_construction(self, fs):
        fs.write("data.json", b'{"name": "test", "value": 42}')
        output = execute_script(to_script('jq "{n: .name, v: .value}" data.json'), fs)
        assert '"n":' in output or '"n": ' in output
        assert '"v":' in output or '"v": ' in output

    def test_array_construction(self, fs):
        fs.write("data.json", b'{"a": 1, "b": 2}')
        # Use [.a] since comma causes shell parsing issues
        output = execute_script(to_script('jq "[.a]" data.json'), fs)
        assert "1" in output

    def test_empty_array(self, fs):
        output = execute_script(to_script('jq -n "[]"'), fs)
        assert output.strip() == "[]"

    def test_empty_object(self, fs):
        output = execute_script(to_script('jq -n "{}"'), fs)
        assert output.strip() == "{}"


# =============================================================================
# Operator tests
# =============================================================================


class TestOperators:
    def test_arithmetic(self, fs):
        output = execute_script(to_script('jq -n "2 + 3"'), fs)
        assert output.strip() == "5"
        output = execute_script(to_script('jq -n "10 - 4"'), fs)
        assert output.strip() == "6"
        output = execute_script(to_script('jq -n "3 * 4"'), fs)
        assert output.strip() == "12"
        output = execute_script(to_script('jq -n "15 / 3"'), fs)
        assert output.strip() == "5.0"

    def test_comparison(self, fs):
        output = execute_script(to_script('jq -n "1 == 1"'), fs)
        assert output.strip() == "true"
        output = execute_script(to_script('jq -n "1 != 2"'), fs)
        assert output.strip() == "true"
        output = execute_script(to_script('jq -n "1 < 2"'), fs)
        assert output.strip() == "true"

    def test_boolean_and(self, fs):
        output = execute_script(to_script('jq -n "true and true"'), fs)
        assert output.strip() == "true"
        output = execute_script(to_script('jq -n "true and false"'), fs)
        assert output.strip() == "false"

    def test_boolean_or(self, fs):
        output = execute_script(to_script('jq -n "false or true"'), fs)
        assert output.strip() == "true"
        output = execute_script(to_script('jq -n "false or false"'), fs)
        assert output.strip() == "false"

    def test_not(self, fs):
        # Use 'not' function instead of pipe syntax due to shell issues
        output = execute_script(to_script("jq -n 'true | not'"), fs)
        assert output.strip() == "false"

    def test_alternative(self, fs):
        fs.write("data.json", b'{"a": null}')
        # Use single quotes to avoid escaping issues
        output = execute_script(to_script("jq '.a // 42' data.json"), fs)
        assert "42" in output

    def test_string_concat(self, fs):
        output = execute_script(to_script('jq -n \'"hello" + " " + "world"\' '), fs)
        assert "hello world" in output


# =============================================================================
# Conditional tests
# =============================================================================


class TestConditional:
    def test_if_then_else(self, fs):
        fs.write("data.json", b'{"x": 5}')
        # Use single quotes and numbers to avoid shell escaping issues
        output = execute_script(
            to_script("jq 'if .x > 3 then 1 else 0 end' data.json"), fs
        )
        assert output.strip() == "1"

    def test_if_then_else_false(self, fs):
        fs.write("data.json", b'{"x": 1}')
        output = execute_script(
            to_script("jq 'if .x > 3 then 1 else 0 end' data.json"), fs
        )
        assert output.strip() == "0"


# =============================================================================
# Error handling tests
# =============================================================================


class TestErrorHandling:
    def test_missing_file(self, fs):
        with pytest.raises(TerminalError) as exc:
            execute_script(to_script('jq "." missing.json'), fs)
        assert "No such file" in str(exc.value)

    def test_invalid_json(self, fs):
        fs.write("bad.json", b"not json")
        with pytest.raises(TerminalError) as exc:
            execute_script(to_script('jq "." bad.json'), fs)
        assert "Invalid JSON" in str(exc.value)

    def test_invalid_filter(self, fs):
        fs.write("data.json", b'{"a": 1}')
        with pytest.raises(TerminalError) as exc:
            execute_script(to_script('jq ".[" data.json'), fs)
        assert "parse error" in str(exc.value)

    def test_type_error(self, fs):
        fs.write("data.json", b'"not an object"')
        with pytest.raises(TerminalError) as exc:
            execute_script(to_script('jq ".foo" data.json'), fs)
        assert "Cannot index" in str(exc.value)

    def test_optional_field_no_error(self, fs):
        fs.write("data.json", b'{"a": 1}')
        # .b? should not error, just return null
        output = execute_script(to_script('jq ".b?" data.json'), fs)
        assert output.strip() == "null"


# =============================================================================
# Pipeline integration tests
# =============================================================================


class TestPipelineIntegration:
    def test_grep_jq_pipeline(self, fs):
        # Simulate a typical workflow: find JSON files, process them
        fs.write("data.json", b'{"items": [{"name": "foo"}, {"name": "bar"}]}')
        output = execute_script(
            to_script('cat data.json | jq -r ".items[] | .name"'), fs
        )
        lines = output.strip().split("\n")
        assert "foo" in lines
        assert "bar" in lines

    def test_jq_wc_pipeline(self, fs):
        fs.write("data.json", b"[1, 2, 3, 4, 5]")
        output = execute_script(to_script('jq ".[]" data.json | wc -l'), fs)
        assert "5" in output
