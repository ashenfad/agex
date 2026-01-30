import ast
import asyncio
from typing import Any, Callable

from agex.agent.base import BaseAgent
from agex.fs.base import FileSystem
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
        session: str = "default",
        package: str = "",
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
        self.resolver = Resolver(agent, session=session)
        self.on_event = on_event
        self.on_token = on_token
        self.main_loop = main_loop
        self.session = session
        self.package = package
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
    eval_timeout_seconds: float | None = None,
    *,
    fs: FileSystem | None = None,
    session: str = "default",
    package: str = "",
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
        eval_timeout_seconds: Optional timeout override. If None, uses agent.eval_timeout_seconds
        fs: Optional filesystem instance to use for file operations (keyword-only)
        session: Session identifier for VFS access (keyword-only)
        package: Package context for relative imports (e.g., "app" for app/main.py)
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
        session=session,
        package=package,
    )

    # Set up filesystem context if fs is provided
    fs_context = None
    if fs is not None:
        from agex.fs import swap_agent_fs_functions, with_fs_context

        # Swap any registered fs functions (like open) with FS-aware wrappers
        # This handles the case where agent.fn(open) registered the real function
        swap_agent_fs_functions(agent)
        fs_context = with_fs_context(fs)

    # Set up network sandbox context - blocks network during agent code evaluation
    from agex.net import deny_network
    from agex.net.patch import install_network_sandbox

    install_network_sandbox()  # Idempotent - installs patches if not already done
    network_context = deny_network()

    try:
        # Enter filesystem context if configured
        if fs_context is not None:
            fs_context.__enter__()

        # Enter network sandbox context
        network_context.__enter__()

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
        # Exit network sandbox context
        network_context.__exit__(None, None, None)
        # Exit filesystem context if we entered it
        if fs_context is not None:
            fs_context.__exit__(None, None, None)


def run_file_in_sandbox(
    agent: BaseAgent,
    file_path: str,
    session: str = "default",
    *,
    eval_timeout_seconds: float | None = None,
    on_event: Callable[[Any], None] | None = None,
    on_token: Callable[[Any], None] | None = None,
    main_loop: asyncio.AbstractEventLoop | None = None,
) -> State:
    """
    Run a file from VFS in the agent's sandbox.

    This is a convenience function for executing code from the virtual filesystem
    using the agent's registered modules, functions, and classes. Useful for
    running user-generated code (e.g., apps) in a sandboxed environment.

    Args:
        agent: The agent providing the execution context and VFS access
        file_path: Path to the file in VFS (e.g., "app/main.py")
        session: Session identifier for state/fs access
        eval_timeout_seconds: Optional timeout override
        on_event: Optional handler to call for each event
        on_token: Optional handler to call for each token
        main_loop: Optional asyncio loop for bridging async calls

    Returns:
        The state after execution

    Raises:
        FileNotFoundError: If the file doesn't exist in VFS
        EvalError: If the code fails to execute

    Example:
        # Create a sandbox with copied registrations
        sandbox = Agent.clone_registrations(
            main_agent,
            name="sandbox",
            state=connect_state(type="versioned", storage="memory"),
        )
        sandbox.module(ui)  # Add UI module to sandbox

        # Run user-generated app code
        run_file_in_sandbox(sandbox, "app/main.py", session_id)
    """
    # Get filesystem and state for this session
    fs = agent.fs(session)
    state = agent.state(session)

    # Read the file from VFS
    if not fs.exists(file_path):
        raise FileNotFoundError(f"File not found in VFS: {file_path}")

    code = fs.read(file_path).decode("utf-8")

    # Derive package from file path for relative import support
    # "app/main.py" -> "app"
    # "app/sub/module.py" -> "app.sub"
    # "main.py" -> ""
    package = ""
    if "/" in file_path:
        dir_path = file_path.rsplit("/", 1)[0]
        package = dir_path.replace("/", ".")

    # Get the underlying filesystem backend for evaluate_program
    backend, _ = agent._get_fs_backend(session)

    # Execute in sandbox
    evaluate_program(
        code,
        agent,
        state,
        eval_timeout_seconds=eval_timeout_seconds,
        fs=backend,
        session=session,
        package=package,
        on_event=on_event,
        on_token=on_token,
        main_loop=main_loop,
    )

    return state
