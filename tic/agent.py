import fnmatch
import inspect
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Iterable, Literal, Union

Pattern = Union[str, Iterable[str], Callable[[str], bool]]
Visibility = Literal["high", "medium", "low"]


class _AgentExit(Exception):
    """Base class for agent exit signals. Should not be caught by user code."""

    pass


@dataclass
class ExitSuccess(_AgentExit):
    """Signal that the agent has completed its task successfully."""

    result: Any


@dataclass
class ExitFail(_AgentExit):
    """Signal that the agent has failed and cannot complete its task."""

    reason: str


@dataclass
class ExitClarify(_AgentExit):
    """Signal that the agent needs clarification from the user."""

    question: str


@dataclass
class Task:
    """A placeholder for a task definition."""

    # For now, we don't need any fields.
    # This will be fleshed out later.
    pass


def task(func):
    """A decorator to mark a function as an agent task."""
    return Task()


@dataclass
class MemberSpec:
    visibility: Visibility | None = None
    docstring: str | None = None
    constructable: bool | None = None


@dataclass
class AttrDescriptor:
    # A descriptor to hold metadata until the class is processed.
    default: Any
    visibility: Visibility


@dataclass
class RegisteredItem:
    visibility: Visibility


@dataclass
class RegisteredFn(RegisteredItem):
    fn: Callable
    docstring: str | None


@dataclass
class RegisteredClass(RegisteredItem):
    """Represents a registered class and its members."""

    cls: type
    constructable: bool
    # 'visibility' on RegisteredItem is the default.
    attrs: dict[str, MemberSpec] = field(default_factory=dict)
    methods: dict[str, MemberSpec] = field(default_factory=dict)


@dataclass
class RegisteredModule(RegisteredItem):
    """Represents a registered module with its selected members."""

    name: str  # The name the agent will use to import it
    module: ModuleType
    fns: dict[str, MemberSpec] = field(default_factory=dict)
    consts: dict[str, MemberSpec] = field(default_factory=dict)
    classes: dict[str, RegisteredClass] = field(default_factory=dict)


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


class Agent:
    def __init__(self, primer: str | None = None):
        self.primer = primer
        self.fn_registry: dict[str, RegisteredFn] = {}
        self.cls_registry: dict[str, RegisteredClass] = {}
        self.importable_modules: dict[str, RegisteredModule] = {}

    def fn(
        self,
        _fn: Callable | None = None,
        *,
        visibility: Visibility = "high",
        docstring: str | None = None,
    ):
        """
        Registers a function with the agent.
        Can be used as a decorator (`@agent.fn`) or a direct call (`agent.fn(...)(...)`).
        """

        def decorator(f: Callable) -> Callable:
            final_doc = docstring if docstring is not None else f.__doc__
            self.fn_registry[f.__name__] = RegisteredFn(
                fn=f, visibility=visibility, docstring=final_doc
            )
            return f

        return decorator(_fn) if _fn else decorator

    def cls(
        self,
        _cls: type | None = None,
        *,
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
            # 1. Generate all possible members
            all_members = {
                name
                for name, member in inspect.getmembers(c)
                if not name.startswith("__") or name == "__init__"
            }.union(getattr(c, "__annotations__", {}))

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

            for name in selected_names:
                config = final_configure.get(name, MemberSpec())
                # If visibility is not specified in config, use class default
                vis = config.visibility or visibility
                doc = config.docstring

                member = getattr(c, name, None)
                if inspect.isroutine(member):
                    final_methods[name] = MemberSpec(visibility=vis, docstring=doc)
                else:
                    final_attrs[name] = MemberSpec(visibility=vis, docstring=doc)

            self.cls_registry[c.__name__] = RegisteredClass(
                cls=c,
                visibility=visibility,
                constructable=constructable,
                attrs=final_attrs,
                methods=final_methods,
            )
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

        self.importable_modules[final_name] = RegisteredModule(
            name=final_name,
            module=mod,
            visibility=visibility,
            fns=mod_fns,
            consts=mod_consts,
            classes=mod_classes,
        )

    def task(self, func):
        """A decorator to mark a function as an agent task."""
        return Task()
