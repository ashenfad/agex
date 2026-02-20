"""
Bridge layer between agex's registration/state system and sblite's Sandbox.

This package translates agex's AgentPolicy and kvit state into
sblite's Policy and namespace dict, then processes the ExecResult
back into agex's state and event system.
"""

from typing import TYPE_CHECKING, Any, Callable

from kvit import Store
from sblite import Sandbox

from .namespace import build_namespace, make_print_handler
from .policy import translate_policy
from .result import handle_result

if TYPE_CHECKING:
    from agex.agent.base import BaseAgent
    from agex.fs.base import FileSystem


def execute_sandboxed(
    program: str,
    agent: "BaseAgent",
    state: Store,
    eval_timeout_seconds: float | None = None,
    *,
    fs: "FileSystem | None" = None,
    session: str = "default",
    on_event: Callable[[Any], None] | None = None,
) -> None:
    """
    Execute agent code in the sblite sandbox.

    This is the bridge equivalent of evaluate_program() — it translates
    agex's registrations and state into sblite's Sandbox.exec() and
    syncs the results back.

    Args:
        program: The Python code to execute
        agent: The agent providing the execution context
        state: The kvit state to execute in
        eval_timeout_seconds: Optional timeout override
        fs: Optional filesystem instance for file operations
        session: Session identifier
        on_event: Optional handler for events
    """
    timeout = (
        eval_timeout_seconds
        if eval_timeout_seconds is not None
        else agent.eval_timeout_seconds
    )

    # 1. Translate policy
    policy = translate_policy(agent, timeout=timeout)

    # 2. Create sandbox with custom print handler
    print_handler = make_print_handler(state, agent.name, on_event)
    sandbox = Sandbox(
        policy, mode="wrapped", filesystem=fs, print_handler=print_handler
    )

    # 3. Build namespace from state + builtins
    namespace, pre_keys = build_namespace(state, agent, agent.name, on_event=on_event)

    # 4. Execute
    result = sandbox.exec(program, namespace=namespace)

    # 5. Handle result (syncs state, re-raises signals)
    handle_result(result, state, agent.name, pre_keys, on_event=on_event)
