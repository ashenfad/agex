import re
from pathlib import Path
from typing import Any, Callable

from agex.agent.base import BaseAgent


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

    # Session IDs are identifiers, not paths — constrain to safe characters
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", session):
        raise ValueError("Invalid session identifier")

    session_root = Path(root) / session
    session_root.mkdir(parents=True, exist_ok=True)
    return str(session_root)


def run_file_in_sandbox(
    agent: BaseAgent,
    file_path: str,
    session: str = "default",
    *,
    eval_timeout_seconds: float | None = None,
    on_event: Callable[[Any], None] | None = None,
) -> Any:
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
    from agex.eval.bridge import execute_sandboxed

    # Get filesystem and state for this session
    fs = agent.fs(session)
    state = agent.state(session)

    # Read the file from VFS
    if not fs.exists(file_path):
        raise FileNotFoundError(f"File not found in VFS: {file_path}")

    code = fs.read(file_path).decode("utf-8")

    # Get the underlying filesystem backend
    backend, _ = agent._get_fs_backend(session)

    # Normalize file_path for relative import resolution
    normalized = file_path if file_path.startswith("/") else f"/{file_path}"

    # Execute in sandbox
    execute_sandboxed(
        code,
        agent,
        state,
        eval_timeout_seconds=eval_timeout_seconds,
        fs=backend,
        session=session,
        on_event=on_event,
        file_path=normalized,
    )

    return state
