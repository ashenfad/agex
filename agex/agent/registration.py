import inspect
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, TypeVar, overload

from agex.agent.base import BaseAgent, resolve_agent
from agex.agent.datatypes import (
    RESERVED_NAMES,
    MemberSpec,
    Pattern,
    Visibility,
)
from agex.agent.policy.resolve import make_predicate
from agex.agent.utils import get_instance_attributes_from_init
from agex.eval.functions import UserFunction
from agex.eval.objects import AgexModule

if TYPE_CHECKING:
    from agex.host.dependencies import Dependencies

T = TypeVar("T", bound=type)
F = TypeVar("F", bound=Callable[..., Any])


class RegistrationMixin(BaseAgent):
    @overload
    def fn(
        self,
        _fn: F,
        *,
        name: str | None = None,
        visibility: Visibility = "high",
        docstring: str | None = None,
    ) -> F: ...

    @overload
    def fn(
        self,
        _fn: None = None,
        *,
        name: str | None = None,
        visibility: Visibility = "high",
        docstring: str | None = None,
    ) -> Callable[[F], F]: ...

    def fn(
        self,
        _fn: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        visibility: Visibility = "high",
        docstring: str | None = None,
    ) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Registers a function with the agent.
        Can be used as a decorator (`@agent.fn`) or a direct call (`agent.fn(...)`).
        """

        def decorator(f: F) -> F:
            # Check if this is a UserFunction (agent registering function from another agent)

            if isinstance(f, UserFunction):
                # Special case: registering a UserFunction from parent agent
                final_name = name or f.name
                if final_name in RESERVED_NAMES:
                    raise ValueError(
                        f"The name '{final_name}' is reserved and cannot be registered."
                    )

                # Create wrapper that preserves UserFunction call semantics
                def user_function_wrapper(*args, **kwargs):
                    return f(
                        *args, **kwargs
                    )  # UserFunction.__call__ handles agent resolution

                # Preserve metadata from UserFunction
                user_function_wrapper.__name__ = f.name
                user_function_wrapper.__doc__ = f.source_text or "User-defined function"

                # Use provided docstring or fall back to UserFunction source
                final_doc = (
                    docstring
                    if docstring is not None
                    else (f.source_text or user_function_wrapper.__doc__)
                )

                # Register in new policy system
                self._policy.register_fn(
                    func=user_function_wrapper,
                    name=final_name,
                    visibility=visibility,
                    docstring=final_doc,
                )

                self._update_fingerprint()

                # Return the wrapper for consistency
                return user_function_wrapper
            else:
                # Normal case: real Python function
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
                    if sub_agent_host is not None and not isinstance(
                        sub_agent_host, Local
                    ):
                        host_type = type(sub_agent_host).__name__
                        raise ValueError(
                            f"Cannot register task '{final_name}' from a sub-agent with "
                            f"host={host_type}. When using hierarchical agents with Modal, "
                            f"sub-agents must use Local host. The parent's Modal container "
                            f"will execute sub-agent tasks locally."
                        )

                if final_name in RESERVED_NAMES:
                    raise ValueError(
                        f"The name '{final_name}' is reserved and cannot be registered."
                    )
                # Track module for lazy dependency resolution
                if hasattr(f, "__module__"):
                    self._track_module(f.__module__)

                final_doc = docstring if docstring is not None else f.__doc__
                self._policy.register_fn(
                    func=f,
                    name=final_name,
                    visibility=visibility,
                    docstring=final_doc,
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
            )

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
        exception_mappings: dict[type, type] | None = None,
        recursive: bool = False,
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
            )
            # Track module for lazy dependency resolution
            self._track_module(obj.__name__ if hasattr(obj, "__name__") else None)
            self._update_fingerprint()
            return None

        # Check if this is an AgexModule (agent registering module from another agent)

        if isinstance(obj, AgexModule):
            # Special case: inherit from parent agent via policy 'inherited' namespace
            parent_agent = resolve_agent(obj.agent_fingerprint)
            parent_ns = parent_agent._policy.namespaces.get(obj.name)  # type: ignore[attr-defined]
            if parent_ns is not None:
                from agex.agent.policy.datatypes import Namespace

                final_name = name or obj.name
                if final_name in RESERVED_NAMES:
                    raise ValueError(
                        f"The name '{final_name}' is reserved and cannot be registered."
                    )
                child_ns = Namespace(
                    name=final_name,
                    kind="inherited",
                    module=parent_ns.module,
                    visibility=visibility,
                    include=include,
                    exclude=tuple(exclude) if isinstance(exclude, list) else exclude,
                    configure={},
                    recursive=False,
                    parent=parent_ns,
                )
                self._policy.namespaces[final_name] = child_ns
            self._update_fingerprint()

        # Check if we're dealing with a module or an instance
        elif isinstance(obj, ModuleType):
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
                exception_mappings=exception_mappings,
            )
            # Store the live instance in the host registry for runtime access
            self._host_object_registry[name] = obj
            # Track module for lazy dependency resolution
            if hasattr(obj, "__module__"):
                self._track_module(obj.__module__)
            self._update_fingerprint()

    # NOTE: `_handle_agex_module_inheritance` removed. Inheritance is handled lazily
    # via a child policy namespace of kind "inherited" created in `module()`.

    def _track_module(self, module_name: str | None) -> None:
        """Record a module name for lazy dependency resolution (fast, no package lookup)."""
        if module_name:
            self._tracked_modules.add(module_name)

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

        # Cache packages_distributions() result for this computation
        try:
            pkg_map = metadata.packages_distributions()
        except Exception:
            pkg_map = {}

        for module_name in self._tracked_modules:
            # Skip standard library and built-ins
            if (
                module_name in sys.stdlib_module_names
                or module_name in sys.builtin_module_names
            ):
                continue

            # Get top-level package name
            top_level = module_name.split(".")[0]

            try:
                # Map import name to distribution name (e.g. sklearn -> scikit-learn)
                dist_names = pkg_map.get(top_level)
                if dist_names:
                    for pkg in dist_names:
                        try:
                            version = metadata.version(pkg)
                            packages.add(f"{pkg}=={version}")
                        except metadata.PackageNotFoundError:
                            pass
            except Exception:
                pass

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

        deps = Dependencies(
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
            agex_version=metadata.version("agex"),
            packages=sorted(list(packages)),
        )
        self._cached_dependencies = deps
        return deps
