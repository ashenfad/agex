import inspect
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, TypeVar, overload

from agex.agent.base import BaseAgent
from agex.agent.datatypes import (
    RESERVED_NAMES,
    MemberSpec,
    Pattern,
    Visibility,
)
from agex.agent.policy.resolve import make_predicate
from agex.agent.utils import get_instance_attributes_from_init

if TYPE_CHECKING:
    from agex.host.dependencies import Dependencies

T = TypeVar("T", bound=type)
F = TypeVar("F", bound=Callable[..., Any])


def _cached_packages_distributions() -> dict[str, list[str]]:
    """Return packages_distributions(), cached for the lifetime of the process."""
    global _pkg_distributions_cache
    if _pkg_distributions_cache is None:
        from importlib import metadata

        try:
            _pkg_distributions_cache = metadata.packages_distributions()
        except Exception:
            _pkg_distributions_cache = {}
    return _pkg_distributions_cache


_pkg_distributions_cache: dict[str, list[str]] | None = None


def _is_local_module(module_name: str) -> bool:
    """
    Determine whether a module should be treated as local to the project/workspace.

    In this context, "local" means code that lives in the current repository/workspace,
    rather than third-party packages installed into site-packages or the standard library.

    A module is considered "local" if:
    - It has no distribution metadata (i.e., is not a packaged dependency)
    - Or it is installed in editable mode from a location outside site-packages

    Local modules need to be added to Modal images via add_local_python_source.
    """
    import os
    import site
    import sys
    from importlib import metadata

    if not module_name:
        return False

    top_level = module_name.split(".")[0]

    # Skip stdlib and builtins
    if top_level in sys.stdlib_module_names or top_level in sys.builtin_module_names:
        return False

    # Skip agex itself - it gets installed on Modal
    if top_level == "agex":
        return False

    # Check if module has PyPI distribution metadata
    try:
        metadata.version(top_level)
        # Has version = pip installed, but might be editable
        # Check if it's in site-packages
        try:
            mod = sys.modules.get(top_level)
            if mod is None:
                import importlib

                mod = importlib.import_module(top_level)

            if hasattr(mod, "__file__") and mod.__file__:
                module_path = os.path.abspath(mod.__file__)
                site_paths = site.getsitepackages()
                user_site = site.getusersitepackages()
                if user_site:
                    site_paths = site_paths + [user_site]

                # If module is NOT in site-packages, it's a local/editable install
                in_site = any(
                    module_path.startswith(os.path.abspath(sp))
                    for sp in site_paths
                    if sp
                )
                return not in_site
        except Exception:
            # Broad catch needed: dynamic import/path operations can fail in many ways
            # (ImportError, AttributeError, OSError, etc.) depending on module state
            pass
        return False  # Has metadata, assume it's from PyPI
    except metadata.PackageNotFoundError:
        # No distribution metadata - could be local OR an internal module (like _pytest)
        # Check if the module is in site-packages
        try:
            mod = sys.modules.get(top_level)
            if mod is None:
                import importlib

                mod = importlib.import_module(top_level)

            if hasattr(mod, "__file__") and mod.__file__:
                module_path = os.path.abspath(mod.__file__)
                site_paths = site.getsitepackages()
                user_site = site.getusersitepackages()
                if user_site:
                    site_paths = site_paths + [user_site]

                # If module IS in site-packages, it's an internal PyPI package module
                in_site = any(
                    module_path.startswith(os.path.abspath(sp))
                    for sp in site_paths
                    if sp
                )
                if in_site:
                    return False  # In site-packages = not local
        except Exception:
            # Broad catch needed: dynamic import/path operations can fail in many ways
            # (ImportError, AttributeError, OSError, etc.) depending on module state
            pass

        # No metadata and not in site-packages = definitely local
        return True


class RegistrationMixin(BaseAgent):
    @overload
    def fn(
        self,
        _fn: F,
        *,
        name: str | None = None,
        visibility: Visibility = "high",
        docstring: str | None = None,
        host_fs_access: bool = False,
        network_access: bool = False,
    ) -> F: ...

    @overload
    def fn(
        self,
        _fn: None = None,
        *,
        name: str | None = None,
        visibility: Visibility = "high",
        docstring: str | None = None,
        host_fs_access: bool = False,
        network_access: bool = False,
    ) -> Callable[[F], F]: ...

    def fn(
        self,
        _fn: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        visibility: Visibility = "high",
        docstring: str | None = None,
        host_fs_access: bool = False,
        network_access: bool = False,
    ) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Registers a function with the agent.
        Can be used as a decorator (`@agent.fn`) or a direct call (`agent.fn(...)`).
        """

        def decorator(f: F) -> F:
            final_name = name or f.__name__

            # Check if this function is already a task on THIS agent
            # This would indicate API confusion - tasks don't need .fn() registration
            owning_agent = getattr(f, "__agex_agent__", None)
            if owning_agent is self:
                raise ValueError(
                    f"Cannot register '{final_name}' as a capability on the same "
                    f"agent that owns it as a task. Task functions are automatically "
                    f"available to their agent—use @agent.fn only when registering "
                    f"a task from a different agent as a callable capability."
                )

            # Check for nested Modal hosts - not supported
            # If this is a task from another agent with a Modal host, reject it
            if owning_agent is not None:
                from agex.host.local import Local

                sub_agent_host = getattr(owning_agent, "_host", None)
                if sub_agent_host is not None and not isinstance(sub_agent_host, Local):
                    host_type = type(sub_agent_host).__name__
                    raise ValueError(
                        f"Cannot register task '{final_name}' from a sub-agent with "
                        f"host={host_type}. When using hierarchical agents with Modal, "
                        f"sub-agents must use Local host. The parent's Modal container "
                        f"will execute sub-agent tasks locally."
                    )

                # Check for sub-agent state when parent uses Modal
                # Sub-agents with persistent state aren't supported on Modal yet
                # because Modal Dict/Volume for sub-agents can't be provisioned
                parent_host = getattr(self, "_host", None)
                if parent_host is not None and not isinstance(parent_host, Local):
                    sub_agent_state = getattr(owning_agent, "_state_config", None)
                    if sub_agent_state is not None:
                        state_type = getattr(sub_agent_state, "type", "ephemeral")
                        if state_type != "ephemeral":
                            parent_host_type = type(parent_host).__name__
                            raise ValueError(
                                f"Cannot register task '{final_name}' from a sub-agent "
                                f"with state=connect_state(type='{state_type}', ...). "
                                f"When the parent agent uses {parent_host_type} host, "
                                f"sub-agents cannot have persistent state. "
                                f"Use state=None (ephemeral) for sub-agents, or run "
                                f"the parent locally."
                            )

                # Check for nested process isolation — daemon processes can't fork
                parent_isolation = getattr(self, "isolation", "none")
                sub_isolation = getattr(owning_agent, "isolation", "none")
                if parent_isolation != "none" and sub_isolation != "none":
                    raise ValueError(
                        f"Cannot register task '{final_name}' from a sub-agent "
                        f"with isolation='{sub_isolation}' on a parent agent with "
                        f"isolation='{parent_isolation}'. Process-isolated agents "
                        f"cannot nest because daemon processes cannot fork children.\n"
                        f"Fix: Set isolation='none' on the parent (orchestrator) agent, "
                        f"or on the sub-agent."
                    )

            if final_name in RESERVED_NAMES:
                raise ValueError(
                    f"The name '{final_name}' is reserved and cannot be registered."
                )
            # Track module for lazy dependency resolution
            if hasattr(f, "__module__"):
                self._track_module(f.__module__)

            final_doc = docstring if docstring is not None else f.__doc__
            # Preserve network_access from TaskWrappers (sub-agent tasks need network for LLM calls)
            effective_network_access = network_access or getattr(
                f, "network_access", False
            )
            self._policy.register_fn(
                func=f,
                name=final_name,
                visibility=visibility,
                docstring=final_doc,
                host_fs_access=host_fs_access,
                network_access=effective_network_access,
            )

            self._update_fingerprint()

            # Mark as fn-decorated for dual-decorator validation (allow multiple fn decorators)
            # Only set attributes if the function allows it (built-ins don't)
            try:
                if not hasattr(f, "__agent_fn_owners__"):
                    f.__agent_fn_owners__ = []
                f.__agent_fn_owners__.append(self)
                f.__is_agent_fn__ = True  # Keep this for task decorator to detect
            except (AttributeError, TypeError):
                # Built-in functions and some other types don't allow setting attributes
                # This is fine - they can't be task-decorated anyway, so no validation needed
                pass

            return f

        return decorator(_fn) if _fn else decorator

    @overload
    def cls(
        self,
        _cls: T,
        *,
        name: str | None = None,
        visibility: Visibility = "high",
        constructable: bool = True,
        include: Pattern | None = "*",
        exclude: Pattern | None = "_*",
        configure: dict[str, MemberSpec] | None = None,
        host_fs_access: bool = False,
        network_access: bool = False,
    ) -> T: ...

    @overload
    def cls(
        self,
        *,
        name: str | None = None,
        visibility: Visibility = "high",
        constructable: bool = True,
        include: Pattern | None = "*",
        exclude: Pattern | None = "_*",
        configure: dict[str, MemberSpec] | None = None,
        host_fs_access: bool = False,
        network_access: bool = False,
    ) -> Callable[[T], T]: ...

    def cls(
        self,
        _cls: T | None = None,
        *,
        name: str | None = None,
        visibility: Visibility = "high",
        constructable: bool = True,
        include: Pattern | None = "*",
        exclude: Pattern | None = "_*",
        configure: dict[str, MemberSpec] | None = None,
        host_fs_access: bool = False,
        network_access: bool = False,
    ) -> T | Callable[[T], T]:
        """
        Registers a class with the agent.
        Can be used as a decorator (`@agent.cls`) or a direct call (`agent.cls(MyClass)`).
        """
        final_configure = configure or {}

        def decorator(c: T) -> T:
            final_name = name or c.__name__
            if final_name in RESERVED_NAMES:
                raise ValueError(
                    f"The name '{final_name}' is reserved and cannot be registered."
                )

            # 1. Generate all possible members
            all_members = {
                name
                for name, member in inspect.getmembers(c)
                if not name.startswith("__") or name == "__init__"
            }.union(getattr(c, "__annotations__", {}))

            # Add instance attributes from __init__ method when using wildcard patterns
            if include == "*" or (isinstance(include, str) and "*" in include):
                instance_attrs = get_instance_attributes_from_init(c)
                all_members.update(instance_attrs)

            if isinstance(include, (list, set)):
                # Explicitly add the included names, as they might be instance attributes
                # not found by inspect.getmembers on the class.
                all_members.update(include)

            # 2. Filter members based on include/exclude patterns
            include_pred = make_predicate(include)
            exclude_pred = make_predicate(exclude)
            selected_names = {
                name
                for name in all_members
                if include_pred(name) and not exclude_pred(name)
            }

            # 3. Create MemberSpec objects and apply configurations
            final_attrs: dict[str, MemberSpec] = {}
            final_methods: dict[str, MemberSpec] = {}

            # Handle __init__ separately based on `constructable` flag
            if constructable:
                selected_names.add("__init__")
            elif "__init__" in selected_names:
                selected_names.remove("__init__")

            for member_name in selected_names:
                config = final_configure.get(member_name, MemberSpec())
                vis = config.visibility or visibility
                doc = config.docstring

                # Check if the member is a method/routine on the class
                if hasattr(c, member_name) and inspect.isroutine(
                    getattr(c, member_name)
                ):
                    final_methods[member_name] = MemberSpec(
                        visibility=vis, docstring=doc
                    )
                # If it's not a method, and it was in the include list, treat it as a data attribute
                else:
                    final_attrs[member_name] = MemberSpec(visibility=vis, docstring=doc)

            sec_final_configure = {
                k: MemberSpec(
                    visibility=(v.visibility if v is not None else None),
                    docstring=(v.docstring if v is not None else None),
                    constructable=(v.constructable if v is not None else None),
                )
                for k, v in (final_configure or {}).items()
            }

            # If constructable, explicitly add __init__ to configure with dotted key
            # so it bypasses exclude patterns in policy resolution
            if constructable and "__init__" in final_methods:
                dotted_key = f"{c.__name__}.__init__"
                sec_final_configure[dotted_key] = MemberSpec(
                    visibility=final_methods["__init__"].visibility,
                    docstring=final_methods["__init__"].docstring,
                )

            # Track module for lazy dependency resolution
            self._track_module(c.__module__)

            self._policy.register_cls(
                cls=c,
                name=final_name,
                visibility=visibility,
                constructable=constructable,
                include=include,
                exclude=exclude,
                configure=sec_final_configure,
                host_fs_access=host_fs_access,
                network_access=network_access,
            )

            # Attach host_fs_access and network_access to the class itself
            # so instance methods can check them
            try:
                c.__agex_host_fs_access__ = host_fs_access
                c.__agex_network_access__ = network_access
            except (AttributeError, TypeError):
                # Can't set attributes on built-in types, skip
                pass

            self._update_fingerprint()
            return c

        if _cls is None:
            return decorator
        return decorator(_cls)

    def module(
        self,
        obj: Any,
        *,
        name: str | None = None,
        visibility: Visibility = "high",
        include: Pattern | None = "*",
        exclude: Pattern | None = ["_*", "*._*"],
        configure: dict[str, MemberSpec] | None = None,
        recursive: bool = False,
        host_fs_access: bool = False,
        network_access: bool = False,
    ) -> None:
        """
        Registers a module or instance object and its members with the agent.
        """
        if recursive:
            if not isinstance(obj, ModuleType):
                raise TypeError(
                    "The 'recursive' option is only supported for module registration, not for class instances."
                )
            # Validate reserved names
            final_name = name or (obj.__name__ if isinstance(obj, ModuleType) else None)
            if final_name in RESERVED_NAMES:
                raise ValueError(
                    f"The name '{final_name}' is reserved and cannot be registered."
                )
            sec_configure = {
                k: MemberSpec(
                    visibility=v.visibility,
                    docstring=v.docstring,
                    constructable=v.constructable,
                )
                for k, v in (configure or {}).items()
            }
            self._policy.register_module(
                name=name,
                module=obj,
                visibility=visibility,
                include=include,
                exclude=tuple(exclude) if isinstance(exclude, list) else exclude,
                configure=sec_configure,
                recursive=True,
                host_fs_access=host_fs_access,
                network_access=network_access,
            )
            # Track module for lazy dependency resolution
            self._track_module(obj.__name__ if hasattr(obj, "__name__") else None)
            self._update_fingerprint()
            return None

        # Check if we're dealing with a module or an instance
        if isinstance(obj, ModuleType):
            # Validate reserved names
            final_name = name or obj.__name__
            if final_name in RESERVED_NAMES:
                raise ValueError(
                    f"The name '{final_name}' is reserved and cannot be registered."
                )
            sec_configure = {
                k: MemberSpec(
                    visibility=(v.visibility if v is not None else None),
                    docstring=(v.docstring if v is not None else None),
                    constructable=(v.constructable if v is not None else None),
                )
                for k, v in (configure or {}).items()
            }
            self._policy.register_module(
                name=name,
                module=obj,
                visibility=visibility,
                include=include,
                exclude=tuple(exclude) if isinstance(exclude, list) else exclude,
                configure=sec_configure,
                recursive=False,
                host_fs_access=host_fs_access,
                network_access=network_access,
            )
            # Track module for lazy dependency resolution
            self._track_module(obj.__name__ if hasattr(obj, "__name__") else None)
            self._update_fingerprint()
        else:
            sec_configure = {
                k: MemberSpec(
                    visibility=(v.visibility if v is not None else None),
                    docstring=(v.docstring if v is not None else None),
                    constructable=(v.constructable if v is not None else None),
                )
                for k, v in (configure or {}).items()
            }
            if name is None:
                raise TypeError(
                    "The 'name' parameter is required when registering an instance object."
                )
            if name in RESERVED_NAMES:
                raise ValueError(
                    f"The name '{name}' is reserved and cannot be registered."
                )

            # Check if agent uses remote host - live objects can't be serialized
            from agex.host.local import Local

            if not isinstance(self._host, Local):
                raise ValueError(
                    f"Cannot register live object '{name}' on agent with remote host "
                    f"({type(self._host).__name__}). Live objects cannot be serialized "
                    f"for remote execution. Use agent.fn() to register functions or "
                    f"agent.cls() to register classes instead."
                )

            self._policy.register_instance(
                name=name if name is not None else "",
                obj=obj,
                visibility=visibility,
                include=include,
                exclude=tuple(exclude) if isinstance(exclude, list) else exclude,
                configure=sec_configure,
                host_fs_access=host_fs_access,
                network_access=network_access,
            )
            # Store the live instance in the host registry for runtime access
            self._host_object_registry[name] = obj
            # Track module for lazy dependency resolution
            if hasattr(obj, "__module__"):
                self._track_module(obj.__module__)
            self._update_fingerprint()

    @overload
    def terminal(
        self,
        _handler: F,
        *,
        name: str | None = None,
        visibility: Visibility = "high",
        docstring: str | None = None,
    ) -> F: ...

    @overload
    def terminal(
        self,
        _handler: None = None,
        *,
        name: str | None = None,
        visibility: Visibility = "high",
        docstring: str | None = None,
    ) -> Callable[[F], F]: ...

    def terminal(
        self,
        _handler: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        visibility: Visibility = "high",
        docstring: str | None = None,
    ) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a terminal command for ``terminal_action``.

        The handler receives a :class:`~agex.terminal.TerminalContext`
        per invocation and returns ``None`` (success, exit code 0) or
        a :class:`~agex.terminal.CommandResult` (with ``exit_code`` /
        ``stderr`` set).

        Can be used as a decorator (``@agent.terminal``) or as a
        direct call (``agent.terminal(handler, name=...)``).

        Args:
            _handler: The handler callable.  When omitted, this method
                returns a decorator (decorator-factory pattern).
            name: Override the command name.  Defaults to
                ``handler.__name__``.  Cannot be a name in
                :data:`~agex.terminal.RESERVED_TERMINAL_NAMES`
                (currently ``{"python"}``).
            visibility: Surfaces the command in the agent's primer.
                ``"high"`` (default): name + docstring; ``"medium"``:
                name only; ``"low"``: not in primer (rely on ``--help``
                or skill markdown).  The command works regardless —
                visibility only controls primer placement.
            docstring: Override ``handler.__doc__`` for the primer.

        Returns:
            The handler unchanged (decorator-friendly).

        Raises:
            ValueError: If ``name`` collides with a reserved
                terminal-command name.

        Example::

            @agent.terminal
            def esbuild(ctx):
                '''Bundle JS files.  Run `esbuild --help` for options.'''
                ...
        """
        from agex.terminal import (
            RESERVED_TERMINAL_NAMES,
            TerminalCommandRegistration,
        )

        def decorator(handler: F) -> F:
            final_name = name or getattr(handler, "__name__", None)
            if not final_name:
                raise ValueError(
                    "agent.terminal: handler has no __name__; pass name= explicitly."
                )
            if final_name in RESERVED_TERMINAL_NAMES:
                raise ValueError(
                    f"'{final_name}' is reserved (agex-internal command); "
                    f"use a different name."
                )
            final_doc = docstring if docstring is not None else handler.__doc__
            self._terminal_commands[final_name] = TerminalCommandRegistration(
                name=final_name,
                handler=handler,
                kind="simple",
                visibility=visibility,
                docstring=final_doc,
            )
            return handler

        return decorator(_handler) if _handler is not None else decorator

    def _terminal_command_factory(
        self,
        name: str,
        factory: Callable[..., Any],
        *,
        visibility: Visibility = "high",
        docstring: str | None = None,
    ) -> None:
        """Register a terminal command via a factory closure (internal).

        For commands that need per-action agex runtime context (state,
        vfs).  The factory is called once per ``terminal_action`` with
        a fresh :class:`~agex.terminal.TerminalRuntime` and returns a
        termish-shape :class:`~agex.terminal.CommandFunc` (a callable
        taking :class:`~agex.terminal.CommandContext` and returning
        ``CommandResult | None``).

        **Internal API.** Currently used only by ``register_git`` for
        access to the agent's per-action ``Staged`` and VFS internals.
        Will be promoted to a public ``agent.terminal_factory`` method
        if real downstream cases emerge that need per-action runtime
        context — for now, public registrations should use
        :meth:`terminal` and reach for runtime values via closures
        over the agent at registration time when needed.

        Args:
            name: The command name.  Required (factory has no
                ``__name__`` semantics for the command itself).
                Cannot be in :data:`~agex.terminal.RESERVED_TERMINAL_NAMES`.
            factory: A callable taking a
                :class:`~agex.terminal.TerminalRuntime` and returning a
                :class:`~agex.terminal.CommandFunc`.
            visibility: Same semantics as :meth:`terminal`.
            docstring: Description for the primer.  When omitted,
                falls back to ``factory.__doc__``.

        Raises:
            ValueError: If ``name`` collides with a reserved
                terminal-command name.
        """
        from agex.terminal import (
            RESERVED_TERMINAL_NAMES,
            TerminalCommandRegistration,
        )

        if name in RESERVED_TERMINAL_NAMES:
            raise ValueError(
                f"'{name}' is reserved (agex-internal command); use a different name."
            )
        final_doc = docstring if docstring is not None else factory.__doc__
        self._terminal_commands[name] = TerminalCommandRegistration(
            name=name,
            handler=factory,
            kind="factory",
            visibility=visibility,
            docstring=final_doc,
        )

    def _track_module(self, module_name: str | None) -> None:
        """Record a module name for lazy dependency resolution (fast, no package lookup)."""
        if module_name:
            self._tracked_modules.add(module_name)

    def _get_installed_optional_deps(self, package_name: str) -> set[str]:
        """
        Find optional dependencies of a package that are installed locally.

        For packages with extras (e.g., `calgebra[google]`), this detects which
        optional dependencies are actually installed in the environment so they
        can be included in remote execution images.

        Args:
            package_name: Distribution name (e.g., "calgebra", not import name)

        Returns:
            Set of installed optional deps as "package==version" strings
        """
        import re
        from importlib import metadata

        installed_optionals: set[str] = set()

        try:
            reqs = metadata.requires(package_name) or []
        except metadata.PackageNotFoundError:
            return installed_optionals

        for req in reqs:
            # Look for extras markers like: 'gcsa; extra == "google"'
            if "extra ==" in req or "extra==" in req:
                # Extract the package name (before the semicolon)
                dep_spec = req.split(";")[0].strip()
                # Remove version specifiers: "gcsa>=1.0" -> "gcsa"
                dep_name = re.split(r"[<>=!~\[]", dep_spec)[0].strip()

                if not dep_name:
                    continue

                # Check if it's actually installed
                try:
                    version = metadata.version(dep_name)
                    installed_optionals.add(f"{dep_name}=={version}")
                except metadata.PackageNotFoundError:
                    pass  # Not installed, skip

        return installed_optionals

    @property
    def dependencies(self) -> "Dependencies":
        """
        Software dependencies inferred from registered functions, modules, and classes.

        Returns:
            Dependencies object containing python version, agex version, and packages.

        Note: Dependencies are computed lazily on first access and cached.
        The cache is invalidated when registrations change.
        """
        import sys
        from importlib import metadata

        from agex.host.dependencies import Dependencies

        # Return cached if available
        if self._cached_dependencies is not None:
            return self._cached_dependencies

        # Build packages list from tracked modules (expensive, but only once)
        packages: set[str] = set()
        local_packages: set[str] = set()  # Local packages for add_local_python_source
        # Track distribution names for optional dep lookup
        distribution_names: set[str] = set()

        pkg_map = _cached_packages_distributions()

        for module_name in self._tracked_modules:
            # Skip standard library and built-ins
            if (
                module_name in sys.stdlib_module_names
                or module_name in sys.builtin_module_names
            ):
                continue

            # Get top-level package name
            top_level = module_name.split(".")[0]

            # Check if this is a local package first
            if _is_local_module(module_name):
                local_packages.add(top_level)
                continue

            try:
                # Map import name to distribution name (e.g. sklearn -> scikit-learn)
                dist_names = pkg_map.get(top_level)
                if dist_names:
                    for pkg in dist_names:
                        try:
                            version = metadata.version(pkg)
                            packages.add(f"{pkg}=={version}")
                            distribution_names.add(pkg)
                        except metadata.PackageNotFoundError:
                            pass
                else:
                    # Fallback for Python 3.10 where packages_distributions() may be incomplete
                    # Try using the module name directly as package name (works for most packages)
                    try:
                        version = metadata.version(top_level)
                        packages.add(f"{top_level}=={version}")
                        distribution_names.add(top_level)
                    except metadata.PackageNotFoundError:
                        pass
            except Exception:
                pass

        # Detect installed optional dependencies for each tracked package
        # This ensures extras like `calgebra[google]` have their optional deps
        # (e.g., gcsa) included if they're installed locally
        for dist_name in distribution_names:
            optional_deps = self._get_installed_optional_deps(dist_name)
            packages.update(optional_deps)

        # Collect dependencies from sub-agents (hierarchical agents)
        # When an agent uses sub-agents via @agent.fn @sub_agent.task,
        # the sub-agent gets serialized in the closure and needs its deps too
        sub_agents = set()
        for namespace in self._policy.namespaces.values():
            # Check fn_objects for task functions from other agents
            for fn in namespace.fn_objects.values():
                # Check if this function is a task from another agent
                owning_agent = getattr(fn, "__agex_agent__", None)
                if owning_agent is not None and owning_agent is not self:
                    sub_agents.add(owning_agent)

        # Union sub-agent dependencies
        for sub_agent in sub_agents:
            # Recursively get sub-agent deps (they might have their own sub-agents)
            if hasattr(sub_agent, "dependencies"):
                sub_deps = sub_agent.dependencies
                packages.update(sub_deps.packages)
                local_packages.update(sub_deps.local_packages)

        deps = Dependencies(
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
            agex_version=metadata.version("agex"),
            packages=sorted(list(packages)),
            local_packages=sorted(list(local_packages)),
        )
        self._cached_dependencies = deps
        return deps
