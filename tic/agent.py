import fnmatch
import inspect
from dataclasses import dataclass, field, fields, is_dataclass
from types import ModuleType
from typing import Any, Callable, Iterable, Literal, Union

Selector = Union[str, Iterable[str], Callable[[str], bool]]
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
class MemberInfo:
    visibility: Visibility


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
    attrs: dict[str, MemberInfo] = field(default_factory=dict)
    methods: dict[str, MemberInfo] = field(default_factory=dict)


@dataclass
class RegisteredModule(RegisteredItem):
    """Represents a registered module with its selected members."""

    name: str  # The name the agent will use to import it
    module: ModuleType
    fns: dict[str, MemberInfo] = field(default_factory=dict)
    consts: dict[str, MemberInfo] = field(default_factory=dict)
    classes: dict[str, RegisteredClass] = field(default_factory=dict)


def _create_predicate(selector: Selector | None) -> Callable[[str], bool]:
    """Creates a predicate function from a selector."""
    if selector is None:
        return lambda _: False  # Select nothing
    if callable(selector):
        return selector
    if isinstance(selector, str):
        return lambda name: fnmatch.fnmatch(name, selector)
    if isinstance(selector, (list, set)):
        return lambda name: any(fnmatch.fnmatch(name, pattern) for pattern in selector)
    raise TypeError(f"Unsupported selector type: {type(selector)}")


class Select:
    """Provides common selection predicates."""

    @staticmethod
    def non_private(name: str) -> bool:
        """
        Selects items that do not start with a single underscore and are not
        "dunder" methods/attributes.
        """
        return not name.startswith("_") and not (
            name.startswith("__") and name.endswith("__")
        )


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
        attrs: Selector | None = None,
        methods: Selector | None = None,
        overrides: dict[str, dict[str, Any]] | None = None,
    ):
        """
        Registers a class with the agent.
        Can be used as a decorator (`@agent.cls`) or a direct call (`agent.cls(MyClass)`).
        """
        final_overrides = overrides or {}

        def decorator(c: type) -> type:
            final_attrs: dict[str, MemberInfo] = {}
            final_methods: dict[str, MemberInfo] = {}
            default_visibility = visibility

            # 1. Populate with defaults based on broad selectors
            # Special case for dataclasses where attrs is not specified
            if attrs is None and is_dataclass(c):
                for f in fields(c):
                    final_attrs[f.name] = MemberInfo(visibility=default_visibility)
            else:  # General attribute selector
                attr_pred = _create_predicate(attrs)
                all_possible_attrs = {
                    name
                    for name, member in inspect.getmembers(c)
                    if not inspect.isroutine(member)
                }.union(getattr(c, "__annotations__", {}))
                for name in all_possible_attrs:
                    if attr_pred(name):
                        final_attrs[name] = MemberInfo(visibility=default_visibility)

            meth_pred = _create_predicate(methods)
            for name, member in inspect.getmembers(c):
                if inspect.isroutine(member) and meth_pred(name):
                    final_methods[name] = MemberInfo(visibility=default_visibility)

            # 2. Apply overrides
            for name, override_settings in final_overrides.items():
                # Don't allow overrides to implicitly expose private members,
                # unless they were already explicitly selected by the main selectors.
                if (
                    name.startswith("_")
                    and name not in final_methods
                    and name not in final_attrs
                ):
                    continue

                member = getattr(c, name, None)
                if not member:
                    continue  # Skip if the attribute doesn't exist on the class

                vis = override_settings.get("visibility", default_visibility)

                if inspect.isroutine(member):
                    final_methods[name] = MemberInfo(visibility=vis)
                else:
                    final_attrs[name] = MemberInfo(visibility=vis)

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
        fns: Selector | None = Select.non_private,
        consts: Selector | None = Select.non_private,
        classes: Selector | None = Select.non_private,
        class_attrs: Selector | None = Select.non_private,
        class_methods: Selector | None = Select.non_private,
        overrides: dict[str, dict[str, Any]] | None = None,
    ):
        """
        Registers a module, making it available for the agent to import.
        """
        module_name = name or mod.__name__
        if module_name == "__main__":
            raise ValueError(
                "Cannot infer module name for '__main__'. Please provide 'name'."
            )
        final_overrides = overrides or {}

        fn_pred = _create_predicate(fns)
        const_pred = _create_predicate(consts)
        class_pred = _create_predicate(classes)

        selected_fns: dict[str, MemberInfo] = {}
        selected_consts: dict[str, MemberInfo] = {}
        selected_classes: dict[str, RegisteredClass] = {}

        # 1. Broad selection pass
        for member_name, member in inspect.getmembers(mod):
            if inspect.isroutine(member):
                if fn_pred(member_name):
                    selected_fns[member_name] = MemberInfo(visibility=visibility)
            elif inspect.isclass(member):
                if class_pred(member_name):
                    # We create the RegisteredClass here, applying nested overrides.
                    class_vis = final_overrides.get(member_name, {}).get(
                        "visibility", visibility
                    )
                    nested_overrides = final_overrides.get(member_name, {}).get(
                        "overrides", None
                    )

                    # Use the main `cls` registration logic by calling it directly.
                    # This avoids re-implementing the logic and ensures consistency.
                    # We pass `_cls` to call it directly instead of as a decorator.
                    temp_agent = Agent()
                    temp_agent.cls(
                        member,
                        visibility=class_vis,
                        constructable=True,
                        attrs=class_attrs,
                        methods=class_methods,
                        overrides=nested_overrides,
                    )
                    selected_classes[member_name] = temp_agent.cls_registry[member_name]

            elif (
                not callable(member)
                and not inspect.ismodule(member)
                and const_pred(member_name)
            ):
                selected_consts[member_name] = MemberInfo(visibility=visibility)

        # 2. Apply top-level overrides
        for member_name, override_settings in final_overrides.items():
            if member_name in selected_classes:
                continue  # Class overrides are handled during their creation

            vis = override_settings.get("visibility", visibility)
            member = getattr(mod, member_name, None)
            if inspect.isroutine(member):
                selected_fns[member_name] = MemberInfo(visibility=vis)
            elif not callable(member) and not inspect.ismodule(member):
                selected_consts[member_name] = MemberInfo(visibility=vis)

        self.importable_modules[module_name] = RegisteredModule(
            name=module_name,
            module=mod,
            visibility=visibility,
            fns=selected_fns,
            consts=selected_consts,
            classes=selected_classes,
        )

    def task(self, func):
        pass
