"""
Modal execution host for serverless agent deployment.

This module provides the Modal host implementation that executes agent tasks
on Modal's serverless infrastructure. Functions are created programmatically
from the agent's dependencies and host configuration — no pre-deployment required.
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from agex.host.base import Host
from agex.host.local import Local
from agex.state import Live, Versioned
from agex.state.kv.modal_dict import ModalDict

if TYPE_CHECKING:
    from agex.agent.base import BaseAgent
    from agex.host.dependencies import Dependencies
    from agex.state import State
    from agex.state.config import StateConfig


# ---- Cloudpickle serialization helpers ----
# Used for streaming results between Modal container and client


def _serialize_result(result: Any) -> dict:
    """Serialize a result for streaming with cloudpickle."""
    import cloudpickle

    return {
        "type": "result",
        "data": cloudpickle.dumps(result),
        "encoding": "cloudpickle",
    }


def _deserialize_result(msg: dict) -> Any:
    """Deserialize a result message, handling cloudpickle encoding."""
    if msg.get("encoding") == "cloudpickle":
        import cloudpickle

        return cloudpickle.loads(msg["data"])
    return msg["data"]


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
            stream_queue.put(_serialize_result(result))

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
            "Use type='ephemeral' for fresh state or type='versioned' for persistence."
        )

    if storage and storage not in ("memory", "disk"):
        raise ValueError(
            f"Modal host supports storage='memory' or 'disk', got '{storage}'"
        )

    # For disk storage, require path to determine Volume name
    if storage == "disk":
        path = getattr(config, "path", None)
        if not path:
            raise ValueError(
                "Disk storage on Modal requires 'path' parameter to name the Volume. "
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

    def resolve_state(
        self, config: "StateConfig | None", session: str, fingerprint: str = ""
    ) -> "State":
        """
        Resolve state for Modal container execution.

        Storage semantics:
          - memory: Disk + ModalDict (7-day TTL on inactive keys)
          - disk: Disk + ModalDict + WriteBehind(Volume) (forever)

        Names are auto-generated from fingerprint+session if no path provided.
        """
        import re
        import shutil
        from pathlib import Path

        from agex.state import Live, Versioned
        from agex.state.kv import Disk
        from agex.state.kv.composite import Composite
        from agex.state.kv.modal_dict import ModalDict

        if config is None:
            return Live()

        state_type = getattr(config, "type", "ephemeral")
        storage = getattr(config, "storage", "memory")

        if state_type == "ephemeral":
            return Live()

        if state_type == "live":
            return Live()

        if state_type != "versioned":
            return Live()

        # --- Versioned state ---

        config_path = getattr(config, "path", None) or ""

        def sanitize_name(name: str) -> str:
            """Sanitize name for Modal Dict/Volume naming."""
            # Expand ~ in paths
            name = name.replace("~", "home")
            # Replace path separators and invalid chars with dots
            name = re.sub(r"[^a-zA-Z0-9._-]+", ".", name)
            # Remove leading/trailing dots
            name = name.strip(".")
            return name or "state"

        # Determine base name
        if config_path:
            base_name = sanitize_name(config_path)
        else:
            # Auto-generate from fingerprint
            fp_short = fingerprint[:12] if fingerprint else "default"
            base_name = f"agex.{fp_short}"

        # Session-scoped names
        dict_name = f"{base_name}.{session}"
        cache_dir = Path(f"/tmp/agex-cache/{dict_name}")

        # Create layers
        source = ModalDict(name=dict_name, prefix=base_name)

        if storage == "disk":
            # Three-tier: Disk → ModalDict → Volume
            from agex.state.kv.modal_volume import Volume

            volume_name = base_name
            volume = Volume(volume_name, prefix=session)

            # Sentinel check on Volume (authoritative tier)
            sentinel_key = "__agex_sentinel__"
            try:
                if sentinel_key not in volume:
                    # Volume was cleared - cascade clear to ModalDict and Disk
                    source.clear()
                    if cache_dir.exists():
                        shutil.rmtree(cache_dir)
                    volume.set(sentinel_key, b"1")
            except Exception:
                pass

            cache = Disk(str(cache_dir))
            kv = Composite([cache, source, volume])
            return Versioned(store=kv)

        else:
            # Two-tier: Disk → ModalDict (memory storage)
            # Sentinel check on ModalDict (authoritative tier)
            sentinel_key = "__agex_sentinel__"
            try:
                if sentinel_key not in source:
                    # Dict was cleared - clear Disk cache
                    if cache_dir.exists():
                        shutil.rmtree(cache_dir)
                    source.set(sentinel_key, b"1")
            except Exception:
                pass

            cache = Disk(str(cache_dir))
            kv = Composite([cache, source])
            return Versioned(store=kv)

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
        fingerprint = getattr(agent, "fingerprint", None)
        if not fingerprint:
            # Compute fingerprint if not already set (e.g., after deserialization)
            from agex.agent.fingerprint import compute_agent_fingerprint_from_policy

            fingerprint = compute_agent_fingerprint_from_policy(agent)
        state = self.resolve_state(agent._state_config, session, fingerprint)

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
        fingerprint = getattr(agent, "fingerprint", None)
        if not fingerprint:
            # Compute fingerprint if not already set (e.g., after deserialization)
            from agex.agent.fingerprint import compute_agent_fingerprint_from_policy

            fingerprint = compute_agent_fingerprint_from_policy(agent)
        state = self.resolve_state(agent._state_config, session, fingerprint)

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

    def _build_function(
        self, deps: "Dependencies", state_config: "StateConfig | None" = None
    ) -> Any:
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

        # Add local packages to the image via add_local_python_source
        # IMPORTANT: This must come AFTER all pip_install calls
        # Modal mounts these at startup rather than embedding in image layer
        if deps.local_packages:
            for pkg_name in deps.local_packages:
                image = image.add_local_python_source(pkg_name)

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

        # Mount Volume if disk storage is used
        if state_config is not None:
            storage = getattr(state_config, "storage", None)
            config_path = getattr(state_config, "path", None) or ""
            if storage == "disk" and config_path:
                import re

                # Sanitize path for volume name (same logic as resolve_state)
                name = config_path.replace("~", "home")
                name = re.sub(r"[^a-zA-Z0-9._-]+", ".", name)
                name = name.strip(".") or "state"
                volume_name = name

                volume = modal.Volume.from_name(volume_name, create_if_missing=True)
                fn_kwargs["volumes"] = {"/vol": volume}

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
            import hashlib

            import modal

            # Try to get existing fingerprint or compute it
            fingerprint = getattr(agent, "fingerprint", None)
            if not fingerprint:
                from agex.agent.fingerprint import compute_agent_fingerprint_from_policy

                fingerprint = compute_agent_fingerprint_from_policy(agent)

            # Sanitize name: alphanum/dash/period/underscore, < 64 chars
            # Format: agex-{clean_name}-{combined_hash}
            clean_name = "".join(c if c.isalnum() else "-" for c in agent.name).lower()

            # Build hash components: fingerprint + deps
            hash_input = f"{fingerprint}:{deps.id}"

            # If disk storage, include volume object_id in hash
            # This ensures new app when volume is recreated (new ID)
            state_config = getattr(agent, "_state_config", None)
            if state_config is not None:
                import re

                storage = getattr(state_config, "storage", None)
                config_path = getattr(state_config, "path", None) or ""
                if storage == "disk" and config_path:
                    name = config_path.replace("~", "home")
                    name = re.sub(r"[^a-zA-Z0-9._-]+", ".", name)
                    name = name.strip(".") or "state"
                    volume_name = name

                    try:
                        volume = modal.Volume.from_name(
                            volume_name, create_if_missing=True
                        )
                        volume.hydrate()  # Required to fetch object_id from server
                        volume_id = volume.object_id
                        if volume_id:
                            hash_input = f"{hash_input}:{volume_id}"
                    except Exception:
                        pass  # Volume lookup failed, proceed without volume ID

            combined_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

            # Truncate components to fit within 63 chars
            # "agex-" (5) + name (max 40) + "-" (1) + hash (16) = 62
            clean_name = clean_name[:40]

            self.app = f"agex-{clean_name}-{combined_hash}"

        if self._runner_fn is None:
            # Get state config from agent for volume mounting
            state_config = getattr(agent, "_state_config", None)
            self._build_function(deps, state_config)

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

        result = None
        for msg in gen:
            msg_type = msg.get("type")
            if msg_type == "event" and on_event:
                on_event(msg["data"])
            elif msg_type == "token" and on_token:
                on_token(msg["data"])
            elif msg_type == "result":
                result = _deserialize_result(msg)
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
        except RuntimeError as e:
            # Volume attachment error (e.g., volume was deleted and recreated)
            # Redeploy to mount the new volume
            if "not attached" in str(e) or "volumes={" in str(e):
                old_app = self.app
                # Force rebuild by clearing cached function and app name
                self._runner_fn = None
                self._current_deps_id = None
                self._modal_app = None
                self.app = None  # Force new app name to avoid stale deployment

                # Rebuild with state config (for volume mounting) and deploy
                self._ensure_function(agent)
                self._modal_app.deploy()

                print(
                    f"Volume attachment error. Redeployed from '{old_app}' to '{self.app}'"
                )

                # Explicitly lookup from the NEW app (not cached reference)
                runner = modal.Function.from_name(self.app, "_agex_runner")
                gen = run_gen(runner)
                result = self._process_messages(gen, on_event, on_token)
            else:
                raise

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

    def state(
        self,
        config: "StateConfig | None",
        session: str,
        fingerprint: str = "",
    ) -> "State":
        """
        Get state for client-side access.

        This allows callers to access the same Modal Dict/Volume used by
        remote task execution, enabling operations like cancel, rollback,
        and inspection.

        Note: Requires Modal SDK to be installed and authenticated on the client.
        """

        if config is None:
            return Live()

        state_type = config.type
        storage = config.storage

        if state_type == "ephemeral":
            return Live()

        if state_type != "versioned":
            # Live/other types rejected at validation - fallback for safety
            return Live()

        # --- Versioned state ---

        config_path = config.path or ""

        def sanitize_name(name: str) -> str:
            """Sanitize name for Modal Dict/Volume naming."""
            name = name.replace("~", "home")
            name = re.sub(r"[^a-zA-Z0-9._-]+", ".", name)
            name = name.strip(".")
            return name or "state"

        # Determine base name (same logic as ModalLocal.resolve_state)
        if config_path:
            base_name = sanitize_name(config_path)
        else:
            fp_short = fingerprint[:12] if fingerprint else "default"
            base_name = f"agex.{fp_short}"

        # Session-scoped names
        dict_name = f"{base_name}.{session}"

        # Connect to Modal Dict (no local disk cache for client-side access)
        source = ModalDict(name=dict_name, prefix=base_name)

        if storage == "disk":
            # Also connect to Volume for disk storage
            from agex.state.kv.composite import Composite
            from agex.state.kv.modal_volume import Volume

            volume_name = base_name
            volume = Volume(volume_name, prefix=session)
            kv = Composite([source, volume])
        else:
            kv = source

        # Create Versioned and verify state exists
        versioned = Versioned(store=kv)

        # Check that state has been initialized (has history)
        try:
            history = versioned.history()
            if not history:
                raise ValueError(
                    f"No state found for session '{session}'. "
                    f"Run a task first to initialize state."
                )
        except Exception as e:
            if "No state found" in str(e):
                raise
            raise ValueError(
                f"Could not access Modal state for session '{session}': {e}"
            ) from e

        return versioned
