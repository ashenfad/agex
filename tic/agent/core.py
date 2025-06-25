import fnmatch
import inspect
from types import ModuleType
from typing import Callable, Dict

from .datatypes import (
    RESERVED_NAMES,
    MemberSpec,
    Pattern,
    RegisteredClass,
    RegisteredFn,
    RegisteredModule,
    Visibility,
)
from .fingerprint import compute_agent_fingerprint

# Global registry mapping fingerprints to agents
_AGENT_REGISTRY: Dict[str, "Agent"] = {}


def _create_predicate(pattern: Pattern | None) -> Callable[[str], bool]:
    """Creates a predicate function from a pattern."""
    if pattern is None:
        return lambda _: False  # Select nothing
    if callable(pattern):
        return pattern
    if isinstance(pattern, str):
        return lambda name: fnmatch.fnmatch(name, pattern)
    if isinstance(pattern, (list, set)):
        return lambda name: any(fnmatch.fnmatch(name, p) for p in pattern)
    raise TypeError(f"Unsupported pattern type: {type(pattern)}")


def register_agent(agent: "Agent") -> str:
    """
    Register an agent in the global registry.

    Returns the agent's fingerprint.
    """
    fingerprint = compute_agent_fingerprint(
        agent.primer, agent.fn_registry, agent.cls_registry, agent.importable_modules
    )
    _AGENT_REGISTRY[fingerprint] = agent
    return fingerprint


def resolve_agent(fingerprint: str) -> "Agent":
    """
    Resolve an agent by its fingerprint.

    Raises RuntimeError if no matching agent is found.
    """
    agent = _AGENT_REGISTRY.get(fingerprint)
    if not agent:
        available = list(_AGENT_REGISTRY.keys())
        raise RuntimeError(
            f"No agent found with fingerprint '{fingerprint[:8]}...'. "
            f"Available fingerprints: {[fp[:8] + '...' for fp in available]}"
        )
    return agent


def clear_agent_registry() -> None:
    """Clear the global registry. Primarily for testing."""
    global _AGENT_REGISTRY
    _AGENT_REGISTRY = {}


class Agent:
    def __init__(self, primer: str | None = None, timeout_seconds: float = 5.0):
        self.primer = primer
        self.timeout_seconds = timeout_seconds
        self.fn_registry: dict[str, RegisteredFn] = {}
        self.cls_registry: dict[str, RegisteredClass] = {}
        self.cls_registry_by_type: dict[type, RegisteredClass] = {}
        self.importable_modules: dict[str, RegisteredModule] = {}

        # Auto-register this agent
        self.fingerprint = register_agent(self)

    def _update_fingerprint(self):
        """Update the fingerprint after registration changes."""
        self.fingerprint = register_agent(self)

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
            final_name = name or f.__name__
            if final_name in RESERVED_NAMES:
                raise ValueError(
                    f"The name '{final_name}' is reserved and cannot be registered."
                )
            final_doc = docstring if docstring is not None else f.__doc__
            self.fn_registry[final_name] = RegisteredFn(
                fn=f, visibility=visibility, docstring=final_doc
            )
            self._update_fingerprint()
            return f

        return decorator(_fn) if _fn else decorator

    def cls(
        self,
        _cls: type | None = None,
        *,
        name: str | None = None,
        visibility: Visibility = "high",
        constructable: bool = True,
        include: Pattern | None = "*",
        exclude: Pattern | None = "_*",
        configure: dict[str, MemberSpec] | None = None,
    ):
        """
        Registers a class with the agent.
        Can be used as a decorator (`@agent.cls`) or a direct call (`agent.cls(MyClass)`).
        """
        final_configure = configure or {}

        def decorator(c: type) -> type:
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

            if isinstance(include, (list, set)):
                # Explicitly add the included names, as they might be instance attributes
                # not found by inspect.getmembers on the class.
                all_members.update(include)

            # 2. Filter members based on include/exclude patterns
            include_pred = _create_predicate(include)
            exclude_pred = _create_predicate(exclude)
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

            # Create the spec for the current class
            spec = RegisteredClass(
                cls=c,
                visibility=visibility,
                constructable=constructable,
                attrs=final_attrs,
                methods=final_methods,
            )

            # Inherit from parent specs if they exist
            for parent in c.__bases__:
                # We can only inherit from registered classes
                parent_spec = self.cls_registry_by_type.get(parent)
                if parent_spec:
                    spec.attrs.update(parent_spec.attrs)
                    spec.methods.update(parent_spec.methods)

            self.cls_registry[final_name] = spec
            self.cls_registry_by_type[c] = spec
            self._update_fingerprint()
            return c

        return decorator(_cls) if _cls else decorator

    def module(
        self,
        mod: ModuleType,
        *,
        name: str | None = None,
        visibility: Visibility = "high",
        include: Pattern | None = "*",
        exclude: Pattern | None = "_*",
        configure: dict[str, MemberSpec] | None = None,
    ):
        """
        Registers a module and its members with the agent.
        """
        final_name = name or mod.__name__.split(".")[-1]
        if final_name in RESERVED_NAMES:
            raise ValueError(
                f"The name '{final_name}' is reserved and cannot be registered."
            )
        final_configure = configure or {}

        # 1. Generate all possible members with dot-notation for class members
        all_members = set()
        for member_name, member in inspect.getmembers(mod):
            if member_name.startswith("@"):
                continue
            all_members.add(member_name)
            if inspect.isclass(member):
                for class_member_name, _ in inspect.getmembers(member):
                    if (
                        not class_member_name.startswith("_")
                        or class_member_name == "__init__"
                    ):
                        all_members.add(f"{member_name}.{class_member_name}")

        # 2. Filter members based on include/exclude patterns
        include_pred = _create_predicate(include)
        exclude_pred = _create_predicate(exclude)
        selected_names = {
            name
            for name in all_members
            if include_pred(name) and not exclude_pred(name)
        }

        # 3. Process selected members and apply configurations
        mod_fns: dict[str, MemberSpec] = {}
        mod_consts: dict[str, MemberSpec] = {}
        mod_classes: dict[str, RegisteredClass] = {}

        # Separate class members from top-level members for processing
        top_level_names = {n for n in selected_names if "." not in n}
        class_member_names = selected_names - top_level_names

        for member_name in top_level_names:
            member = getattr(mod, member_name)
            config = final_configure.get(member_name, MemberSpec())
            vis = config.visibility or visibility
            doc = config.docstring

            if inspect.isroutine(member):
                mod_fns[member_name] = MemberSpec(visibility=vis, docstring=doc)
            elif inspect.isclass(member):
                cls_attrs: dict[str, MemberSpec] = {}
                cls_methods: dict[str, MemberSpec] = {}

                # Determine constructability from config, defaulting to True
                cls_is_constructable = (
                    config.constructable if config.constructable is not None else True
                )

                # Get all members for this specific class from the selection
                cls_selected_members = {
                    cm.split(".", 1)[1]
                    for cm in class_member_names
                    if cm.startswith(f"{member_name}.")
                }

                # Also apply the top-level exclude predicate to these members
                exclude_pred = _create_predicate(exclude)
                cls_selected_members = {
                    m
                    for m in cls_selected_members
                    if not exclude_pred(f"{member_name}.{m}")
                }

                # Handle __init__ based on constructability, overriding include/exclude
                if cls_is_constructable:
                    cls_selected_members.add("__init__")
                elif "__init__" in cls_selected_members:
                    cls_selected_members.remove("__init__")

                for short_name in cls_selected_members:
                    class_member = getattr(member, short_name, None)
                    if not class_member:
                        continue

                    cm_config_key = f"{member_name}.{short_name}"
                    cm_config = final_configure.get(cm_config_key, MemberSpec())
                    cm_vis = cm_config.visibility or vis
                    cm_doc = cm_config.docstring

                    if inspect.isroutine(class_member):
                        cls_methods[short_name] = MemberSpec(
                            visibility=cm_vis, docstring=cm_doc
                        )
                    else:
                        cls_attrs[short_name] = MemberSpec(
                            visibility=cm_vis, docstring=cm_doc
                        )

                mod_classes[member_name] = RegisteredClass(
                    cls=member,
                    visibility=vis,
                    constructable=cls_is_constructable,
                    attrs=cls_attrs,
                    methods=cls_methods,
                )
            else:  # It's a constant
                mod_consts[member_name] = MemberSpec(visibility=vis, docstring=doc)

        reg_mod = RegisteredModule(
            name=final_name,
            module=mod,
            visibility=visibility,
            fns=mod_fns,
            consts=mod_consts,
            classes=mod_classes,
        )
        self.importable_modules[final_name] = reg_mod

        # Also add all registered classes to the agent's central by-type registry
        for spec in mod_classes.values():
            self.cls_registry_by_type[spec.cls] = spec

        self._update_fingerprint()

    def task(self, func):
        """A decorator to mark a function as an agent task."""
        from .datatypes import Task

        return Task()
