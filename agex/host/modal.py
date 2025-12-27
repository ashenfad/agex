"""
Modal execution host for serverless agent deployment.

This module provides the Modal host implementation that executes agent tasks
on Modal's serverless infrastructure. Functions are created programmatically
from the agent's dependencies and host configuration — no pre-deployment required.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from agex.host.base import Host
from agex.host.local import Local

if TYPE_CHECKING:
    from agex.agent.base import BaseAgent
    from agex.host.dependencies import Dependencies
    from agex.state import State
    from agex.state.config import StateConfig


def _agex_runner(
    agent_bytes: bytes,
    task_name: str,
    args: tuple,
    kwargs: dict,
    session: str,
):
    """Execute an agex agent task on Modal with real-time streaming."""
    import queue
    import threading

    import cloudpickle

    # Queue for streaming events/tokens/result
    stream_queue: queue.Queue = queue.Queue()

    def run_task():
        """Background thread that runs the task and pushes to queue."""
        try:
            # Deserialize the agent
            agent_instance = cloudpickle.loads(agent_bytes)

            # Rehydrate LLM from config (reads API key from env)
            if agent_instance.llm is None:
                if (
                    hasattr(agent_instance, "_llm_config")
                    and agent_instance._llm_config
                ):
                    from agex.llm import LLM

                    agent_instance.llm = LLM.from_config(agent_instance._llm_config)

            # Use ModalLocal host for execution (handles disk storage → ModalFile)
            agent_instance._host = ModalLocal()

            # Re-register agent for UserFunction resolution
            from agex.agent import register_agent

            agent_instance.fingerprint = register_agent(agent_instance)

            # Find the task function
            task_fn = agent_instance._tasks.get(task_name)
            if task_fn is None:
                stream_queue.put(
                    {"type": "error", "data": f"Task '{task_name}' not found on agent"}
                )
                return

            # Callbacks that push to queue immediately
            def on_event(e):
                stream_queue.put({"type": "event", "data": e})

            def on_token(t):
                stream_queue.put({"type": "token", "data": t})

            # Use the standard execute_task helper (same as HTTP runner)
            from agex.host.runner import execute_task

            result = execute_task(
                agent=agent_instance,
                task_name=task_name,
                args=args,
                kwargs=kwargs,
                session=session,
                state_config=agent_instance._state_config,
                on_event=on_event,
                on_token=on_token,
            )

            # Handle async tasks
            import inspect

            if inspect.isawaitable(result):
                import asyncio

                result = asyncio.run(result)

            # Manually serialize result to prevent Modal from doing it in the IO loop
            stream_queue.put(
                {
                    "type": "result",
                    "data": cloudpickle.dumps(result),
                    "encoding": "cloudpickle",
                }
            )

        except Exception as e:
            import traceback

            stream_queue.put(
                {
                    "type": "error",
                    "data": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                }
            )

        finally:
            # Sentinel to signal completion
            stream_queue.put(None)

    # Start task in background thread
    thread = threading.Thread(target=run_task, daemon=True)
    thread.start()

    # Yield items as they arrive (true streaming)
    while True:
        item = stream_queue.get()
        if item is None:
            break
        yield item

    thread.join()


def _validate_modal_state(config: "StateConfig | None") -> None:
    """
    Validate state config for Modal execution (shared by Modal and ModalLocal).

    Raises:
        ValueError: If state config is incompatible with Modal's serverless model.
    """
    if config is None:
        return  # Ephemeral is always OK

    state_type = getattr(config, "type", "ephemeral")
    storage = getattr(config, "storage", None)

    # Live state doesn't work on Modal - no persistence between invocations
    if state_type == "live":
        raise ValueError(
            "Live state is not supported on Modal (state doesn't persist between task invocations). "
            "Use type='ephemeral' for fresh state or type='versioned' with storage='disk' for persistence."
        )

    # Versioned state with memory storage doesn't work on Modal - memory is reset between invocations
    if state_type == "versioned" and storage == "memory":
        raise ValueError(
            "Versioned state with memory storage is not supported on Modal (memory is reset between invocations). "
            "Use type='ephemeral' for fresh state or type='versioned' with storage='disk' for persistence."
        )

    if storage and storage not in ("memory", "disk"):
        raise ValueError(
            f"Modal host supports storage='memory' or 'disk', got '{storage}'"
        )

    # For disk storage, require path to determine Dict name
    if storage == "disk":
        path = getattr(config, "path", None)
        if not path:
            raise ValueError(
                "Disk storage on Modal requires 'path' parameter to name the Dict. "
                "Example: connect_state(type='versioned', storage='disk', path='my-agent')"
            )


class ModalLocal(Local):
    """
    Local-like host for execution inside Modal containers.

    This host is used by the Modal runner to execute tasks inside the Modal
    container. It's similar to Local but uses ModalDict for disk storage,
    leveraging Modal's native Dict service for fast distributed key-value access.

    Inherits from Local so task dispatch recognizes it as local execution.
    """

    def __init__(self):
        self._session_cache: dict[str, Any] = {}

    def dump_config(self) -> dict[str, Any]:
        """Not used for ModalLocal (internal only)."""
        return {"provider": "modal-local"}

    def validate_state(self, config: "StateConfig | None") -> None:
        """Validate state config."""
        _validate_modal_state(config)

    def resolve_state(self, config: "StateConfig | None", session: str) -> "State":
        """
        Resolve state for Modal container execution.

        For disk storage, uses ModalDict backed by Modal's native Dict service.
        For memory storage, uses standard Memory KV store.
        """
        from agex.state import Live, Versioned
        from agex.state.kv import Memory

        if config is None:
            return Live()

        state_type = getattr(config, "type", "ephemeral")
        storage = getattr(config, "storage", "memory")

        if state_type == "ephemeral":
            return Live()

        if state_type == "versioned":
            if storage == "disk":
                # Use Modal's native Dict service for persistent state
                # Much faster than volume-based storage (single RPC per op)
                import shutil
                from pathlib import Path

                from agex.state.kv import Disk
                from agex.state.kv.modal_dict import ModalDict
                from agex.state.kv.tiered_cache import TieredCache

                # Use path as the Dict name (required for disk storage)
                config_path = getattr(config, "path", None) or ""

                # Sanitize path for Modal Dict name
                # Replace path separators with dots: /tmp/agex/funcy → tmp.agex.funcy
                if config_path:
                    import re

                    # Expand ~ in paths
                    config_path = config_path.replace("~", "home")

                    # Replace path separators and invalid chars with dots
                    config_path = re.sub(r"[^a-zA-Z0-9._-]+", ".", config_path)

                    # Remove leading/trailing dots
                    config_path = config_path.strip(".")

                    # Ensure it's not empty after sanitization
                    if not config_path:
                        config_path = "state"

                # Shard by session: each session gets its own Dict
                # E.g., "tmp.agex.funcy.adam3" instead of shared "tmp.agex.funcy"
                base_name = config_path if config_path else "agex-state"
                dict_name = f"{base_name}.{session}"

                # Use base name as prefix for key namespacing
                prefix = base_name

                # Create source (ModalDict) first
                source = ModalDict(name=dict_name, prefix=prefix)

                # Check for sentinel key to detect Dict recreation
                # If Dict was deleted and recreated, we need to clear stale local cache
                sentinel_key = "__agex_sentinel__"
                cache_dir = Path(f"/tmp/agex-cache/{dict_name}")

                try:
                    # Check if sentinel exists in remote
                    sentinel_exists = sentinel_key in source
                    if not sentinel_exists:
                        # Dict is brand new or was deleted - clear local cache
                        if cache_dir.exists():
                            shutil.rmtree(cache_dir)
                        # Write sentinel to mark this Dict instance
                        source.set(sentinel_key, b"1")
                except Exception:
                    # If check fails, proceed without cache clearing
                    # (network issue, permissions, etc.)
                    pass

                # Compose: Disk (local /tmp, fast) + ModalDict (remote, authoritative)
                # Local cache survives across hot container reuses
                cache = Disk(str(cache_dir))
                kv = TieredCache(cache=cache, source=source)
                return Versioned(store=kv)
            else:
                # Memory storage
                cache_key = f"versioned:{session}"
                if cache_key not in self._session_cache:
                    self._session_cache[cache_key] = Versioned(store=Memory())
                return self._session_cache[cache_key]

        if state_type == "live":
            return Live()

        return Live()

    def execute(
        self,
        agent: "BaseAgent",
        task_name: str,
        args: tuple,
        kwargs: dict,
        session: str,
        on_event: Callable[[Any], None] | None,
        on_token: Callable[[Any], None] | None,
    ) -> Any:
        """Execute task locally within Modal container."""
        state = self.resolve_state(agent._state_config, session)

        task_fn = agent._tasks.get(task_name)
        if task_fn is None:
            raise ValueError(f"Task '{task_name}' not found on agent '{agent.name}'")

        call_kwargs = dict(kwargs)
        call_kwargs["state"] = state
        if on_event is not None:
            call_kwargs["on_event"] = on_event
        if on_token is not None:
            call_kwargs["on_token"] = on_token

        return task_fn(*args, **call_kwargs)

    async def aexecute(
        self,
        agent: "BaseAgent",
        task_name: str,
        args: tuple,
        kwargs: dict,
        session: str,
        on_event: Callable[[Any], None] | None,
        on_token: Callable[[Any], None] | None,
    ) -> Any:
        """Execute async task locally within Modal container."""
        state = self.resolve_state(agent._state_config, session)

        task_fn = agent._tasks.get(task_name)
        if task_fn is None:
            raise ValueError(f"Task '{task_name}' not found on agent '{agent.name}'")

        call_kwargs = dict(kwargs)
        call_kwargs["state"] = state
        if on_event is not None:
            call_kwargs["on_event"] = on_event
        if on_token is not None:
            call_kwargs["on_token"] = on_token

        return await task_fn(*args, **call_kwargs)


class Modal(Host):
    """
    Serverless execution host using Modal.

    Executes agent tasks on Modal's infrastructure with automatic scaling,
    GPU support, and persistent state via Modal Dict. Functions are
    created dynamically from agent dependencies — no pre-deployment needed.

    Args:
        app: Modal app name (required)
        secrets: List of Modal secret names to inject (e.g., ["llm-keys"])
        gpu: GPU type (e.g., "A10G", "T4", "A100")
        memory: Memory in MB
        timeout: Execution timeout in seconds (default: 300)
        **kwargs: Additional options passed to Modal function

    Example:
        from agex import Agent, connect_host, connect_state

        agent = Agent(
            host=connect_host(provider="modal", app="my-agent", secrets=["llm-keys"]),
            state=connect_state(type="versioned", storage="disk", path="my-agent"),
        )

        agent.warmup()  # Builds image, deploys function
        result = my_task(session="user-123")
    """

    def __init__(
        self,
        app: str | None = None,
        secrets: str | list[str] | None = None,
        gpu: str | None = None,
        memory: int | None = None,
        timeout: float = 300.0,
        scaledown_window: int = 300,
        **kwargs: Any,
    ):
        self.app = app

        # Normalize and validate secrets
        if isinstance(secrets, str):
            self.secrets = [secrets]
        else:
            self.secrets = secrets or []

        if not self.secrets:
            raise ValueError(
                "Modal host requires 'secrets' to be provided (e.g., secrets='llm-keys'). "
                "This ensures the agent can access necessary API keys in the remote environment."
            )

        self.gpu = gpu
        self.memory = memory
        self.timeout = timeout
        self.scaledown_window = scaledown_window
        self._options = kwargs

        # Cached Modal app and function
        self._modal_app: Any = None
        self._runner_fn: Any = None
        self._current_deps_id: str | None = None

    def dump_config(self) -> dict[str, Any]:
        """Serialize host configuration for transport."""
        config = {
            "provider": "modal",
            "timeout": self.timeout,
            "scaledown_window": self.scaledown_window,
        }
        if self.app:
            config["app"] = self.app
        if self.secrets:
            config["secrets"] = self.secrets
        if self.gpu:
            config["gpu"] = self.gpu
        if self.memory:
            config["memory"] = self.memory
        if self._options:
            config.update(self._options)
        return config

    def validate_state(self, config: "StateConfig | None") -> None:
        """Validate state config is compatible with Modal host."""
        _validate_modal_state(config)

    def resolve_state(self, config: "StateConfig | None", session: str) -> "State":
        """
        Resolve state for Modal execution.

        Note: This is typically not called during remote execution.
        The runner uses ModalLocal which handles state resolution appropriately.
        """
        from agex.state import Ephemeral

        if config is None:
            return Ephemeral()

        # For remote execution, ModalLocal handles disk/memory storage
        # This fallback handles edge cases like local testing
        raise NotImplementedError(
            "Modal.resolve_state should not be called during normal execution. "
            "Use the task wrapper which routes through ModalLocal on the Modal side."
        )

    def _build_function(self, deps: "Dependencies") -> Any:
        """Build Modal app and function from dependencies."""
        import modal

        # Create app
        self._modal_app = modal.App(self.app)

        # Build image with dependencies
        image = modal.Image.debian_slim(python_version=deps.python_version)

        # Cloudpickle is required for remote execution but optional in core
        image = image.pip_install("cloudpickle")

        # Install agex (handles all core deps via pyproject.toml)
        try:
            import agex

            agex_path = Path(agex.__file__).parent
            repo_path = agex_path.parent
            # Check if this is a development install (source directory with pyproject.toml)
            if (repo_path / "pyproject.toml").exists():
                # Dev mode: copy repo and pip install from it (gets all deps from pyproject.toml)
                image = image.copy_local_dir(str(repo_path), remote_path="/agex_src")
                image = image.run_commands("pip install /agex_src")
            else:
                # Production: install from PyPI (handles all deps)
                image = image.pip_install("agex")
        except Exception:
            # Fallback to PyPI
            image = image.pip_install("agex")

        # Install additional inferred packages (LLM providers, user packages, etc.)
        if deps.packages:
            # Filter out agex itself (already installed above)
            extra_packages = [p for p in deps.packages if not p.startswith("agex")]
            if extra_packages:
                image = image.pip_install(*extra_packages)

        # Configure secrets
        modal_secrets = [modal.Secret.from_name(s) for s in self.secrets]

        # Build function kwargs
        fn_kwargs: dict[str, Any] = {
            "image": image,
            "timeout": int(self.timeout),
            "name": "_agex_runner",  # Explicit name for lookup
            "scaledown_window": self.scaledown_window,
        }
        if modal_secrets:
            fn_kwargs["secrets"] = modal_secrets
        if self.gpu:
            fn_kwargs["gpu"] = self.gpu
        if self.memory:
            fn_kwargs["memory"] = self.memory
        if self._options:
            fn_kwargs.update(self._options)

        # Decorate the runner function with Modal
        # _agex_runner is at module level, so no serialized=True needed
        decorated = self._modal_app.function(**fn_kwargs)(_agex_runner)

        self._runner_fn = decorated
        self._current_deps_id = deps.id

        return decorated

    def warmup(self, deps: "Dependencies") -> None:
        """
        Pre-warm Modal by building the container image and deploying the function.

        This builds an image with the inferred dependencies and eagerly deploys
        to Modal, making the function available for execution.
        """
        # Build function if not already built or if deps changed
        if self._runner_fn is None or self._current_deps_id != deps.id:
            self._build_function(deps)

        # Deploy the app to Modal (persistent)
        self._modal_app.deploy()

    def _get_deps(self, agent: "BaseAgent") -> "Dependencies":
        """Resolve dependencies for the agent."""
        from importlib import metadata

        from agex.agent.registration import RegistrationMixin

        # Calculate source hash if in dev mode (installed from source)
        agex_version = metadata.version("agex")
        source_hash = None

        try:
            import agex

            package_loc = Path(agex.__file__).parent
            if (package_loc.parent / "pyproject.toml").exists():
                # Dev mode: compute hash of agex directory to force rebuilds on code changes
                import hashlib

                hasher = hashlib.md5()
                for p in sorted(package_loc.rglob("*.py")):
                    hasher.update(p.read_bytes())
                source_hash = hasher.hexdigest()[:8]
        except Exception:
            pass

        final_version = f"{agex_version}+{source_hash}" if source_hash else agex_version

        if isinstance(agent, RegistrationMixin):
            deps = agent.dependencies
            # Patch the version to include source hash
            deps.agex_version = final_version
        else:
            import sys
            from importlib import metadata

            from agex.host.dependencies import Dependencies

            deps = Dependencies(
                python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
                agex_version=final_version,
                packages=[],
            )

        # Add LLM provider package based on agent's LLM config
        llm_config = getattr(agent, "_llm_config", None)
        if llm_config:
            provider = llm_config.get("provider", "")
            llm_packages = {
                "gemini": "google-genai",
                "openai": "openai",
                "anthropic": "anthropic",
            }
            if provider in llm_packages:
                pkg = llm_packages[provider]
                if pkg not in deps.packages:
                    deps.packages.append(pkg)
        return deps

    def _ensure_function(self, agent: "BaseAgent") -> Any:
        """Ensure function is built, building if necessary."""
        # Resolve dependencies first to include them in the name hash
        deps = self._get_deps(agent)

        # Lazily set app name if not provided
        if self.app is None:
            # Try to get existing fingerprint or compute it
            fingerprint = getattr(agent, "fingerprint", None)
            if not fingerprint:
                from agex.agent.fingerprint import compute_agent_fingerprint_from_policy

                fingerprint = compute_agent_fingerprint_from_policy(agent)

            # Sanitize name: alphanum/dash/period/underscore, < 64 chars
            # Format: agex-{clean_name}-{combined_hash}
            clean_name = "".join(c if c.isalnum() else "-" for c in agent.name).lower()

            # Combine agent structural fingerprint with dependency hash (includes source code)
            # This ensures we get a new app (new deployment) if either the agent definition
            # OR the libraries/source code change.
            import hashlib

            combined_hash = hashlib.sha256(
                f"{fingerprint}:{deps.id}".encode()
            ).hexdigest()[:16]

            # Truncate components to fit within 63 chars
            # "agex-" (5) + name (max 40) + "-" (1) + hash (16) = 62
            clean_name = clean_name[:40]

            self.app = f"agex-{clean_name}-{combined_hash}"

        if self._runner_fn is None:
            self._build_function(deps)

        return self._runner_fn

    def _process_messages(
        self,
        gen,
        on_event: Callable[[Any], None] | None,
        on_token: Callable[[Any], None] | None,
    ) -> Any:
        """Process message stream from Modal runner.

        Args:
            gen: Generator yielding message dicts from _agex_runner
            on_event: Optional callback for event messages
            on_token: Optional callback for token messages

        Returns:
            The task result

        Raises:
            RuntimeError: If an error message is received
        """
        import cloudpickle

        result = None
        for msg in gen:
            msg_type = msg.get("type")
            if msg_type == "event" and on_event:
                on_event(msg["data"])
            elif msg_type == "token" and on_token:
                on_token(msg["data"])
            elif msg_type == "result":
                # Check for encoding
                if msg.get("encoding") == "cloudpickle":
                    # Deserialize manually serialized result
                    result = cloudpickle.loads(msg["data"])
                else:
                    result = msg["data"]
            elif msg_type == "error":
                raise RuntimeError(msg["data"])
        return result

    def execute(
        self,
        agent: "BaseAgent",
        task_name: str,
        args: tuple,
        kwargs: dict,
        session: str,
        on_event: Callable[[Any], None] | None,
        on_token: Callable[[Any], None] | None,
    ) -> Any:
        """Execute a task synchronously on Modal."""
        import cloudpickle
        import modal

        # Ensure function is built/defined locally
        # (Needed to know app structure even if we lookup remote)
        self._ensure_function(agent)

        # Serialize the agent
        agent_bytes = cloudpickle.dumps(agent)

        # Helper to run the generator
        def run_gen(fn_obj):
            return fn_obj.remote_gen(
                agent_bytes=agent_bytes,
                task_name=task_name,
                args=args,
                kwargs=kwargs,
                session=session,
            )

        # Lookup the deployed function
        runner = modal.Function.from_name(self.app, "_agex_runner")

        try:
            # Execute the task. If function not found, auto-deploy and retry.
            gen = run_gen(runner)
            result = self._process_messages(gen, on_event, on_token)
        except modal.exception.NotFoundError:
            # Auto-deploy
            print(f"Modal app '{self.app}' not found. Deploying...")
            deps = self._get_deps(agent)
            self.warmup(deps)

            # Re-lookup and retry
            runner = modal.Function.from_name(self.app, "_agex_runner")
            gen = run_gen(runner)
            result = self._process_messages(gen, on_event, on_token)

        return result

    async def aexecute(
        self,
        agent: "BaseAgent",
        task_name: str,
        args: tuple,
        kwargs: dict,
        session: str,
        on_event: Callable[[Any], None] | None,
        on_token: Callable[[Any], None] | None,
    ) -> Any:
        """Execute a task asynchronously on Modal."""
        import asyncio

        # Modal's async support - run in executor for now
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.execute(
                agent, task_name, args, kwargs, session, on_event, on_token
            ),
        )
