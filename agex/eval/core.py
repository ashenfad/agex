import ast
import asyncio
from typing import Any, Callable

from agex.agent.base import BaseAgent
from agex.state.core import State

from .base import BaseEvaluator
from .binop import BinOpEvaluator
from .call import CallEvaluator
from .comprehension import ComprehensionEvaluator
from .error import EvalError
from .expressions import ExpressionEvaluator
from .functions import FunctionEvaluator, _ReturnException
from .loops import LoopEvaluator
from .resolver import Resolver
from .statements import StatementEvaluator


class Evaluator(
    CallEvaluator,
    BinOpEvaluator,
    ExpressionEvaluator,
    ComprehensionEvaluator,
    LoopEvaluator,
    FunctionEvaluator,
    StatementEvaluator,
    BaseEvaluator,
):
    """
    The main evaluator, composed of modular mixins from other files.
    """

    def __init__(
        self,
        agent: BaseAgent,
        state: State,
        source_code: str | None = None,
        eval_timeout_seconds: float | None = None,
        start_time: float | None = None,
        sub_agent_time: float = 0.0,
        on_event: Callable[[Any], None] | None = None,
        on_token: Callable[[Any], None] | None = None,
        main_loop: asyncio.AbstractEventLoop | None = None,
    ):
        actual_timeout = (
            eval_timeout_seconds
            if eval_timeout_seconds is not None
            else agent.eval_timeout_seconds
        )
        super().__init__(
            agent,
            state,
            actual_timeout,
            start_time=start_time,
            sub_agent_time=sub_agent_time,
        )
        self.source_code = source_code
        self.resolver = Resolver(agent)
        self.on_event = on_event
        self.on_token = on_token
        self.main_loop = main_loop
        self._with_binding_cleanup: list[tuple[str, Any]] = []

    def visit_Module(self, node: ast.Module):
        """Evaluates a module by visiting each statement in its body."""
        for stmt in node.body:
            self.visit(stmt)

    def cleanup_with_bindings(self):
        """Remove context-manager bound variables that should not persist across iterations."""
        for name, original_value in self._with_binding_cleanup:
            if name not in self.state:
                continue
            try:
                current_value = self.state.get(name)
            except Exception:
                # If accessing the value fails (e.g., unpicklable marker), still attempt removal.
                self.state.remove(name)
                continue

            # only remove if the value is the same (not re-bound)
            if current_value is original_value:
                self.state.remove(name)
        self._with_binding_cleanup.clear()

    def visit_Expr(self, node: ast.Expr):
        """
        Handles expressions that are used as statements.
        The result of the expression is calculated but not stored.
        """
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            # avoid printing for builtins that already print
            if node.value.func.id in ("print", "view_image", "dir", "help"):
                # Still need to visit the call to execute it for its side effect.
                self.visit(node.value)
                return

        self.visit(node.value)


def _get_session_root(root: str, session: str, per_session: bool) -> str:
    """Get session-specific root path if per_session is enabled.

    Args:
        root: Base root directory
        session: Session identifier
        per_session: Whether to create session subdirectories

    Returns:
        Root path (with session subdir if per_session=True)
    """
    if not per_session:
        return root

    from pathlib import Path

    session_root = Path(root) / session
    session_root.mkdir(parents=True, exist_ok=True)
    return str(session_root)


def evaluate_program(
    program: str,
    agent: BaseAgent,
    state: State,
    session: str = "default",
    eval_timeout_seconds: float | None = None,
    on_event: Callable[[Any], None] | None = None,
    on_token: Callable[[Any], None] | None = None,
    main_loop: asyncio.AbstractEventLoop | None = None,
):
    """
    Updates state with the result of running the program. The agent provides
    whitelisted functions and classes valid for the program.

    Args:
        program: The Python code to execute
        agent: The agent providing the execution context
        state: The state to execute in
        session: Session identifier for per-session filesystem isolation
        eval_timeout_seconds: Optional timeout override. If None, uses agent.eval_timeout_seconds
        on_event: Optional handler to call for each event
        on_token: Optional handler to call for each token
        main_loop: Optional asyncio loop for bridging async calls from the thread
    """
    actual_timeout = (
        eval_timeout_seconds
        if eval_timeout_seconds is not None
        else agent.eval_timeout_seconds
    )
    tree = ast.parse(program)
    evaluator = Evaluator(
        agent,
        state,
        source_code=program,
        eval_timeout_seconds=actual_timeout,
        on_event=on_event,
        on_token=on_token,
        main_loop=main_loop,
    )

    # Set up filesystem context if agent has filesystem configured
    fs_context = None
    fs_instance = None
    if hasattr(agent, "_fs_config") and agent._fs_config is not None:
        from agex.fs import (
            IsolatedFS,
            VirtualFS,
            swap_agent_fs_functions,
            with_isolated_fs,
            with_virtual_fs,
        )
        from agex.fs.config import IsolatedFSConfig, VirtualFSConfig

        # Swap any registered fs functions (like open) with FS-aware wrappers
        # This handles the case where agent.fn(open) registered the real function
        swap_agent_fs_functions(agent)

        if isinstance(agent._fs_config, VirtualFSConfig):
            fs_instance = VirtualFS(state)
            fs_context = with_virtual_fs(fs_instance)
        elif isinstance(agent._fs_config, IsolatedFSConfig):
            # For isolated FS, only use state if tracking is enabled
            tracking_state = state if agent._fs_config.tracking else None

            # Get session-specific root if per_session is enabled
            root = _get_session_root(
                agent._fs_config.root, session, agent._fs_config.per_session
            )

            fs_instance = IsolatedFS(root, tracking_state)
            fs_context = with_isolated_fs(fs_instance)

        # Snapshot metadata AFTER setting up context but BEFORE execution
        metadata_before = fs_instance.get_metadata_snapshot() if fs_instance else None
    else:
        metadata_before = None

    try:
        # Enter filesystem context if configured
        if fs_context is not None:
            fs_context.__enter__()

        try:
            evaluator.visit(tree)
        except _ReturnException as e:
            # Convert return statement outside function to a helpful error
            raise EvalError(
                "'return' outside function. You're in an agent environment, not a regular Python function. "
                "Use task_success(result) to complete your task, not return.",
                e.node,
            )
    finally:
        evaluator.cleanup_with_bindings()
        # Exit filesystem context if we entered it
        if fs_context is not None:
            fs_context.__exit__(None, None, None)

        # Emit FileEvent if files changed during execution
        if fs_instance is not None and metadata_before is not None:
            _emit_file_event_if_changed(
                agent, state, fs_instance, metadata_before, on_event
            )


def _emit_file_event_if_changed(
    agent: BaseAgent,
    state: State,
    fs,  # VirtualFS or IsolatedFS instance
    metadata_before: dict,
    on_event: Callable[[Any], None] | None,
) -> None:
    """Emit FileEvent if files changed during agent code execution."""
    from agex.agent.events import FileEvent
    from agex.state.log import add_event_to_log

    metadata_after = fs.get_metadata_snapshot()

    before_paths = set(metadata_before.keys())
    after_paths = set(metadata_after.keys())

    added = list(after_paths - before_paths)
    removed = list(before_paths - after_paths)

    # Modified = same path but different modified_at timestamp
    modified = [
        p
        for p in before_paths & after_paths
        if metadata_before[p].modified_at != metadata_after[p].modified_at
    ]

    # Only emit if something actually changed
    if added or modified or removed:
        event = FileEvent(
            agent_name=agent.name,
            file_source="agent",
            added=added,
            modified=modified,
            removed=removed,
        )
        add_event_to_log(state, event, on_event=on_event)
