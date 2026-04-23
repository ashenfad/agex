"""Conformance tests for :class:`ToolUseWireFormat` as a
:class:`WireFormat` Protocol implementation."""

import pytest

from agex.llm.formats import (
    ToolCallArgDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolUseWireFormat,
    WireFormat,
    XmlWireFormat,
)
from agex.llm.formats.tool_use import TOOL_PYTHON


def test_satisfies_wire_format_protocol():
    fmt = ToolUseWireFormat()
    assert isinstance(fmt, WireFormat)


def test_schema_returns_all_four_tools():
    fmt = ToolUseWireFormat()
    schema = fmt.tool_schema()
    assert schema is not None
    names = {s["name"] for s in schema}
    assert names == {
        "python_action",
        "terminal_action",
        "write_file",
        "edit_file",
    }


def test_primer_non_empty():
    fmt = ToolUseWireFormat()
    primer = fmt.format_primer()
    assert primer and len(primer) > 0


class TestNativeThinkingFlag:
    """When ``native_thinking=True`` the wire format stops asking the
    model to narrate in the schema and trusts the provider to deliver
    thinking / user-facing text as native content blocks (which the
    stream translators capture as ThinkingEmission / TextEmission)."""

    def test_default_keeps_narration_params(self):
        fmt = ToolUseWireFormat()
        schemas = {s["name"]: s for s in fmt.tool_schema() or []}
        py_props = schemas["python_action"]["parameters"]["properties"]
        term_props = schemas["terminal_action"]["parameters"]["properties"]
        assert "thinking" in py_props
        assert "report" in py_props
        assert "thinking" in term_props

    def test_native_thinking_strips_thinking_and_report(self):
        fmt = ToolUseWireFormat(native_thinking=True)
        schemas = {s["name"]: s for s in fmt.tool_schema() or []}
        py_params = schemas["python_action"]["parameters"]
        term_params = schemas["terminal_action"]["parameters"]

        for params in (py_params, term_params):
            assert "thinking" not in params["properties"]
            assert "report" not in params["properties"]
            assert "thinking" not in params["required"]
            assert "report" not in params["required"]

        # File tools are unaffected — they have no narration params
        # to strip.
        assert {"write_file", "edit_file"} <= set(schemas)

    def test_native_thinking_primer_mentions_native_channels(self):
        fmt = ToolUseWireFormat(native_thinking=True)
        primer = fmt.format_primer()
        # The addendum explicitly redirects to native channels.
        assert "native" in primer.lower()
        assert "thinking" in primer.lower()

    def test_default_primer_has_no_native_thinking_addendum(self):
        default_primer = ToolUseWireFormat().format_primer()
        native_primer = ToolUseWireFormat(native_thinking=True).format_primer()
        assert native_primer.startswith(default_primer)
        assert len(native_primer) > len(default_primer)


def test_parse_text_stream_raises_not_implemented():
    fmt = ToolUseWireFormat()
    with pytest.raises(NotImplementedError):
        list(fmt.parse_text_stream(iter(["<PYTHON>x</PYTHON>"])))


def test_xml_format_tool_stream_raises_not_implemented():
    """Symmetric check: XmlWireFormat also refuses the other path."""
    fmt = XmlWireFormat()
    with pytest.raises(NotImplementedError):
        list(fmt.parse_tool_stream(iter([])))


def test_parse_tool_stream_end_to_end():
    """Integration: pick a tool, stream args, verify TokenChunks emitted."""
    import json

    fmt = ToolUseWireFormat()
    args = json.dumps({"title": "t", "thinking": "T", "code": "print(1)"})
    events = [
        ToolCallStart("c1", TOOL_PYTHON),
        ToolCallArgDelta("c1", args),
        ToolCallEnd("c1"),
    ]
    tokens = list(fmt.parse_tool_stream(iter(events)))
    types = {t.type for t in tokens}
    assert {"title", "thinking", "python"}.issubset(types)
    # Each of those types must terminate with a done=True chunk.
    for type_name in ("title", "thinking", "python"):
        same_type = [t for t in tokens if t.type == type_name]
        assert same_type[-1].done is True
