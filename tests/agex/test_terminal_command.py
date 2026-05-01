"""Tests for ``agent.terminal_command`` and ``agent.terminal_command_factory``.

Covers:
- Registration roundtrip (decorator, decorator-factory, direct call)
- Reserved-name collision raises ValueError
- Last-wins override semantics
- ``build_terminal_commands`` wires registrations through to termish
- Factory variant receives per-action ``TerminalRuntime`` and returns
  a working CommandFunc
- Pipeline composition with termish builtins
"""

import io

import pytest
from termish import MemoryFS, execute

from agex import Agent
from agex.agent.loop.event_factories import build_terminal_commands
from agex.terminal import (
    RESERVED_TERMINAL_NAMES,
    CommandResult,
    TerminalCommandRegistration,
    TerminalContext,
)

# ---------------------------------------------------------------------------
# Registration roundtrip
# ---------------------------------------------------------------------------


class TestTerminalCommandRegistration:
    def test_bare_decorator_registers_with_function_name(self):
        a = Agent()

        @a.terminal_command
        def greet(ctx):
            """say hi"""
            ctx.stdout.write("hi\n")

        assert "greet" in a._terminal_commands
        reg = a._terminal_commands["greet"]
        assert isinstance(reg, TerminalCommandRegistration)
        assert reg.name == "greet"
        assert reg.kind == "simple"
        assert reg.visibility == "high"
        assert reg.docstring == "say hi"

    def test_decorator_factory_with_options(self):
        a = Agent()

        @a.terminal_command(name="renamed", visibility="low", docstring="custom")
        def some_handler(ctx):
            """ignored — explicit docstring overrides"""
            pass

        assert "renamed" in a._terminal_commands
        assert "some_handler" not in a._terminal_commands
        reg = a._terminal_commands["renamed"]
        assert reg.visibility == "low"
        assert reg.docstring == "custom"

    def test_direct_call_registration(self):
        a = Agent()

        def my_handler(ctx):
            pass

        result = a.terminal_command(my_handler, name="custom")

        # Direct call returns the handler unchanged (decorator-friendly).
        assert result is my_handler
        assert "custom" in a._terminal_commands
        assert "my_handler" not in a._terminal_commands

    def test_visibility_default_is_high(self):
        a = Agent()

        @a.terminal_command
        def cmd(ctx):
            pass

        assert a._terminal_commands["cmd"].visibility == "high"


# ---------------------------------------------------------------------------
# Factory variant
# ---------------------------------------------------------------------------


class TestTerminalCommandFactory:
    def test_factory_registration(self):
        a = Agent()

        def make_handler(rt):
            """factory docstring"""

            def handler(cmd_ctx):
                cmd_ctx.stdout.write("from factory\n")

            return handler

        a.terminal_command_factory("custom", make_handler)

        reg = a._terminal_commands["custom"]
        assert reg.kind == "factory"
        assert reg.docstring == "factory docstring"
        assert reg.handler is make_handler

    def test_factory_explicit_docstring_overrides(self):
        a = Agent()

        def make_handler(rt):
            """factory docstring"""

            def handler(cmd_ctx):
                pass

            return handler

        a.terminal_command_factory("custom", make_handler, docstring="explicit")

        assert a._terminal_commands["custom"].docstring == "explicit"

    def test_factory_visibility_default_is_high(self):
        a = Agent()

        def factory(rt):
            return lambda cmd_ctx: None

        a.terminal_command_factory("custom", factory)
        assert a._terminal_commands["custom"].visibility == "high"


# ---------------------------------------------------------------------------
# Reserved names
# ---------------------------------------------------------------------------


class TestReservedNames:
    def test_python_is_reserved(self):
        assert "python" in RESERVED_TERMINAL_NAMES

    def test_terminal_command_python_raises(self):
        a = Agent()
        with pytest.raises(ValueError, match="reserved"):

            @a.terminal_command(name="python")
            def cmd(ctx):
                pass

    def test_terminal_command_factory_python_raises(self):
        a = Agent()
        with pytest.raises(ValueError, match="reserved"):
            a.terminal_command_factory("python", lambda rt: lambda c: None)

    def test_termish_builtins_are_NOT_reserved(self):
        # Termish explicitly supports overriding builtins.
        a = Agent()

        @a.terminal_command(name="ls")
        def my_ls(ctx):
            pass

        # No raise; the registration succeeds.
        assert "ls" in a._terminal_commands


# ---------------------------------------------------------------------------
# Override semantics
# ---------------------------------------------------------------------------


class TestOverride:
    def test_user_registrations_last_wins(self):
        a = Agent()

        @a.terminal_command(name="cmd", docstring="first")
        def first(ctx):
            pass

        @a.terminal_command(name="cmd", docstring="second")
        def second(ctx):
            pass

        assert a._terminal_commands["cmd"].docstring == "second"
        assert a._terminal_commands["cmd"].handler is second

    def test_factory_can_replace_simple_registration(self):
        a = Agent()

        @a.terminal_command(name="cmd")
        def simple(ctx):
            pass

        def make(rt):
            return lambda cmd_ctx: None

        a.terminal_command_factory("cmd", make)

        # Factory registration replaces the simple one.
        assert a._terminal_commands["cmd"].kind == "factory"


# ---------------------------------------------------------------------------
# build_terminal_commands wiring
# ---------------------------------------------------------------------------


class TestBuildTerminalCommands:
    def test_python_is_always_present(self):
        a = Agent()
        fs = MemoryFS()
        commands = build_terminal_commands(a, fs)
        assert "python" in commands

    def test_simple_handler_is_wired_through(self):
        a = Agent()

        @a.terminal_command
        def greet(ctx):
            ctx.stdout.write(f"hello {ctx.args[0]}\n")

        fs = MemoryFS()
        commands = build_terminal_commands(a, fs)

        out = execute("greet world", fs, commands=commands)
        assert out == "hello world\n"

    def test_simple_handler_receives_terminal_context(self):
        a = Agent()
        captured = {}

        @a.terminal_command
        def cmd(ctx):
            # Verify ctx is a TerminalContext, not a CommandContext.
            captured["type"] = type(ctx).__name__
            captured["args"] = list(ctx.args)
            captured["fs_is_passed_through"] = ctx.fs is captured.get("expected_fs")

        fs = MemoryFS()
        captured["expected_fs"] = fs
        commands = build_terminal_commands(a, fs)
        execute("cmd a b c", fs, commands=commands)

        assert captured["type"] == "TerminalContext"
        assert captured["args"] == ["a", "b", "c"]
        assert captured["fs_is_passed_through"] is True

    def test_factory_receives_terminal_runtime(self):
        a = Agent()
        captured = {}

        def make_handler(rt):
            captured["rt_type"] = type(rt).__name__
            captured["rt_fs_class"] = type(rt.fs).__name__
            captured["rt_state"] = rt.state
            captured["rt_vfs"] = rt.vfs

            def handler(cmd_ctx):
                pass

            return handler

        a.terminal_command_factory("custom", make_handler)

        fs = MemoryFS()
        # build_terminal_commands invokes the factory eagerly when wiring.
        build_terminal_commands(a, fs, state="STATE", vfs="VFS")

        assert captured["rt_type"] == "TerminalRuntime"
        assert captured["rt_fs_class"] == "MemoryFS"
        assert captured["rt_state"] == "STATE"
        assert captured["rt_vfs"] == "VFS"

    def test_pipeline_composition_with_termish_builtins(self):
        a = Agent()

        @a.terminal_command
        def emit(ctx):
            ctx.stdout.write("alpha\nbeta\ngamma\n")

        fs = MemoryFS()
        commands = build_terminal_commands(a, fs)

        out = execute("emit | grep beta", fs, commands=commands)
        assert out.strip() == "beta"

    def test_user_registration_can_override_termish_builtin(self):
        # termish's `cat` reads files; we override with one that writes a marker.
        a = Agent()

        @a.terminal_command(name="cat")
        def custom_cat(ctx):
            ctx.stdout.write("OVERRIDDEN\n")

        fs = MemoryFS()
        fs.write("/file.txt", b"original content")
        commands = build_terminal_commands(a, fs)

        out = execute("cat /file.txt", fs, commands=commands)
        assert out == "OVERRIDDEN\n"

    def test_handler_can_return_command_result_for_failure(self):
        a = Agent()

        @a.terminal_command
        def failing(ctx):
            return ctx.fail("something broke", exit_code=2)

        fs = MemoryFS()
        commands = build_terminal_commands(a, fs)

        # termish raises on non-zero exit
        from termish.errors import TerminalError

        with pytest.raises(TerminalError, match="something broke"):
            execute("failing", fs, commands=commands)


# ---------------------------------------------------------------------------
# Per-action freshness
# ---------------------------------------------------------------------------


class TestPerActionFreshness:
    """Each ``terminal_action`` rebuilds its commands dict, so factories
    see fresh runtime values per invocation.  Direct test of the
    contract: invoking ``build_terminal_commands`` twice with different
    state should produce factories with the new values."""

    def test_factory_invoked_with_fresh_runtime_each_build(self):
        a = Agent()
        rt_seen: list[str] = []

        def make_handler(rt):
            rt_seen.append(rt.state)

            def handler(cmd_ctx):
                pass

            return handler

        a.terminal_command_factory("custom", make_handler)

        fs = MemoryFS()
        build_terminal_commands(a, fs, state="state-1")
        build_terminal_commands(a, fs, state="state-2")

        assert rt_seen == ["state-1", "state-2"]


# ---------------------------------------------------------------------------
# TerminalContext fail() helper
# ---------------------------------------------------------------------------


class TestTerminalContextFail:
    def test_fail_returns_command_result(self):
        ctx = TerminalContext(
            args=[],
            stdin=io.StringIO(),
            stdout=io.StringIO(),
            fs=MemoryFS(),
        )
        result = ctx.fail("oops")
        assert isinstance(result, CommandResult)
        assert result.exit_code == 1
        assert result.stderr == "oops"

    def test_fail_custom_exit_code(self):
        ctx = TerminalContext(
            args=[],
            stdin=io.StringIO(),
            stdout=io.StringIO(),
            fs=MemoryFS(),
        )
        result = ctx.fail("bad", exit_code=42)
        assert result.exit_code == 42
        assert result.stderr == "bad"
