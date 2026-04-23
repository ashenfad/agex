"""
Python script handler for termish command injection.

Lets agents run Python scripts from ``<TERMINAL>`` blocks::

    <TERMINAL>python helpers/compute.py arg1 arg2</TERMINAL>

Scripts execute in a sandtrap sandbox with the agent's full policy
(registered modules, gates, VFS) but in a **fresh namespace** — no
REPL state leakage and no ``task_*`` bindings.  Task completion still
happens from ``<PYTHON>`` blocks via the REPL.

Usage::

    from agex.python_cli import make_python_handler, register_python

    handler = make_python_handler(agent, fs)
    execute(script, fs, commands={"python": handler})

    # Or register as a skill:
    register_python(agent)
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from termish.context import CommandContext, CommandResult
from termish.errors import TerminalError

if TYPE_CHECKING:
    from monkeyfs import FileSystem
    from termish.errors import CommandFunc

    from agex.agent.base import BaseAgent


def make_python_handler(agent: "BaseAgent", fs: "FileSystem") -> "CommandFunc":
    """Create a ``python`` command handler.

    Ensures ``sys`` is registered on the agent so scripts can use
    ``import sys`` for ``sys.argv``, ``sys.stdin``, etc.

    Args:
        agent: The agent whose policy governs sandbox execution.
        fs: The VFS to read script files from and to make available
            inside the sandbox for imports and file I/O.

    Returns:
        A termish ``CommandFunc`` for the ``python`` command.
    """
    # Ensure sys is importable — scripts need sys.argv, sys.stdin, etc.
    try:
        agent.module(
            sys,
            visibility="low",
            exclude=["exit", "_exit", "settrace", "setprofile", "setrecursionlimit"],
        )
    except (ValueError, TypeError):
        pass  # Already registered or name conflict — fine

    # Flags that are harmless to ignore (no-op in a sandboxed VFS environment)
    _IGNORED_FLAGS = frozenset({"-u", "-B", "-s", "-E", "-O", "-OO"})
    # Flags that take a following argument (consumed and ignored)
    _IGNORED_ARG_FLAGS = frozenset({"-W", "-X"})

    def handler(ctx: CommandContext) -> CommandResult | None:
        args = list(ctx.args)

        # python (no args) — point the agent to <PYTHON> blocks
        if not args:
            ctx.stdout.write(
                f"Python {sys.version.split()[0]} (sandtrap)\n"
                "Use a <PYTHON> block for interactive Python.\n"
            )
            return None

        # Strip known harmless flags before the filename
        script_args_start = 0
        while script_args_start < len(args):
            a = args[script_args_start]
            if a in _IGNORED_FLAGS:
                args.pop(script_args_start)
            elif a in _IGNORED_ARG_FLAGS:
                # Flag + its argument (e.g. -W ignore)
                args.pop(script_args_start)
                if script_args_start < len(args):
                    args.pop(script_args_start)
            elif a == "-i":
                # -i (interactive after script) — ignore, we can't do interactive
                args.pop(script_args_start)
            else:
                break

        if not args:
            ctx.stdout.write(f"Python {sys.version.split()[0]} (sandtrap)\n")
            return None

        # python --version / python -V
        if args[0] in ("--version", "-V"):
            ctx.stdout.write(f"Python {sys.version.split()[0]}\n")
            return None

        # python --help / python -h
        if args[0] in ("--help", "-h"):
            ctx.stdout.write(
                f"Python {sys.version.split()[0]} (sandtrap)\n\n"
                "usage: python [options] file.py [args...]\n"
                "       python -c 'code' [args...]\n\n"
                "Options (accepted, most ignored in sandbox):\n"
                "  -c code    Execute code string\n"
                "  -u         Unbuffered output (ignored)\n"
                "  -B         Don't write .pyc files (ignored)\n"
                "  -W arg     Warning control (ignored)\n"
                "  -i         Interactive after script (ignored)\n"
                "  --version  Print version\n\n"
                "Note: task_success() is only available from <PYTHON> blocks,\n"
                "not from scripts. Import your work in a <PYTHON> block to\n"
                "complete a task.\n"
            )
            return None

        # python -c "code"
        if args[0] == "-c":
            if len(args) < 2:
                raise TerminalError("python: -c requires an argument")
            code = args[1]
            return _run_code(
                code,
                filename="<string>",
                script_args=["-c"] + args[2:],
                agent=agent,
                fs=fs,
                stdin=ctx.stdin,
                stdout=ctx.stdout,
            )

        # python -m module (not supported)
        if args[0] == "-m":
            raise TerminalError(
                "python -m is not supported. "
                "Use `import module` from a <PYTHON> block instead."
            )

        # Unknown flags
        if args[0].startswith("-"):
            raise TerminalError(
                f"python: unknown option: {args[0]}\n"
                "Use `python --help` for available options."
            )

        # python file.py [args...]
        script_path = args[0]
        script_args = args

        # Read the script from the VFS
        try:
            content_bytes = ctx.fs.read(script_path)
        except FileNotFoundError:
            raise TerminalError(
                f"python: can't open file '{script_path}': No such file"
            )
        except Exception as e:
            raise TerminalError(f"python: error reading '{script_path}': {e}")

        code = content_bytes.decode("utf-8", errors="replace")

        return _run_code(
            code,
            filename=script_path,
            script_args=script_args,
            agent=agent,
            fs=fs,
            stdin=ctx.stdin,
            stdout=ctx.stdout,
        )

    return handler


def _run_code(
    code: str,
    *,
    filename: str,
    script_args: list[str],
    agent: "BaseAgent",
    fs: "FileSystem",
    stdin: Any,
    stdout: Any,
) -> CommandResult | None:
    """Execute Python code in a fresh sandtrap sandbox.

    The sandbox uses the agent's policy (registered modules, gates) and
    the agent's VFS.  The namespace is fresh — no REPL state, no task_*
    bindings.
    """
    from sandtrap import sandbox as create_sandbox

    from agex.eval.bridge.policy import translate_policy

    # Build sandbox with agent's policy and VFS
    tick_limit = getattr(agent, "eval_tick_limit", None)
    if tick_limit is not None:
        timeout = 300.0
    else:
        timeout = agent.eval_timeout_seconds

    policy = translate_policy(agent, timeout=timeout, tick_limit=tick_limit)
    sb = create_sandbox(
        policy,
        isolation=agent.isolation,
        mode="wrapped",
        filesystem=fs,
        snapshot_prints=True,
    )

    # Build a minimal script namespace — no task_*, no REPL state
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "__file__": filename,
    }

    # Set sys.argv and sys.stdin for the duration of execution, then restore.
    # sys is a singleton so import sys inside the script sees these values.
    saved_argv = sys.argv
    saved_stdin = sys.stdin
    sys.argv = list(script_args)
    sys.stdin = stdin
    try:
        with sb:
            result = sb.exec(code, namespace=namespace)
    finally:
        sys.argv = saved_argv
        sys.stdin = saved_stdin

    # Write captured stdout (includes print() output when snapshot_prints=True)
    if result.stdout:
        stdout.write(result.stdout)

    # If the script raised an error, surface it
    if result.error is not None:
        # Check for task_* usage — give a helpful error
        if isinstance(result.error, NameError) and any(
            name in str(result.error)
            for name in ("task_success", "task_fail", "task_clarify")
        ):
            raise TerminalError(
                f"{result.error}\n"
                "task_success() and other task control functions are only "
                "available from <PYTHON> blocks. To complete the task, "
                "import your function from this script in a <PYTHON> block "
                "and call task_success() with its result."
            )
        raise TerminalError(f"{filename}: {result.error}")

    return None
