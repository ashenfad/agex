import inspect
from types import ModuleType
from typing import Any, Callable, TypeVar, overload

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

T = TypeVar("T", bound=type)


class RegistrationMixin(BaseAgent):
    def fn(
        self,
        _fn: Callable | None = None,
        *,
        name: str | None = None,
        visibility: Visibility = "high",
        docstring: str | None = None,
    ):
        """
        Registers a function with the agent.
        Can be used as a decorator (`@agent.fn`) or a direct call (`agent.fn(...)`).
        """

        def decorator(f: Callable) -> Callable:
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
                if final_name in RESERVED_NAMES:
                    raise ValueError(
                        f"The name '{final_name}' is reserved and cannot be registered."
                    )
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
    ):
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
            self._update_fingerprint()
            return

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
            return obj  # Return the AgexModule for consistency

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
            self._update_fingerprint()
            return obj
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
            self._update_fingerprint()
            return obj

    # NOTE: `_handle_agex_module_inheritance` removed. Inheritance is handled lazily
    # via a child policy namespace of kind "inherited" created in `module()`.
