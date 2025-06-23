import fnmatch
import inspect
from dataclasses import dataclass, field, fields, is_dataclass
from types import ModuleType
from typing import Any, Callable, Iterable, Literal, Set, Union

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
class RegisteredItem:
    visibility: Visibility


@dataclass
class RegisteredFn(RegisteredItem):
    fn: Callable
    docstring: str | None


@dataclass
class RegisteredClass(RegisteredItem):
    cls: type
    constructable: bool
    allowed_methods: Set[str] = field(default_factory=set)
    allowed_attrs: Set[str] = field(default_factory=set)


@dataclass
class RegisteredModule(RegisteredItem):
    """Represents a registered module with its selected members."""

    module: ModuleType
    as_name: str
    fns: Set[str] = field(default_factory=set)
    consts: Set[str] = field(default_factory=set)
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
        self.module_registry: dict[str, RegisteredModule] = {}

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
    ):
        """
        Registers a class with the agent.
        Can be used as a decorator (`@agent.cls`) or a direct call (`agent.cls(MyClass)`).
        """

        def decorator(c: type) -> type:
            final_attrs: set[str]
            # Special case: if attrs isn't specified, default to dataclass fields.
            if attrs is None and is_dataclass(c):
                final_attrs = {f.name for f in fields(c)}
            else:
                attr_pred = _create_predicate(attrs)
                # Combine attributes from __annotations__ and class members
                annotated_attrs = {name for name in getattr(c, "__annotations__", {})}
                member_attrs = {
                    name
                    for name, member in inspect.getmembers(c)
                    if not callable(member)
                }
                all_possible_attrs = annotated_attrs.union(member_attrs)
                final_attrs = {name for name in all_possible_attrs if attr_pred(name)}

            meth_pred = _create_predicate(methods)
            final_methods = {
                name
                for name, member in inspect.getmembers(c)
                if callable(member) and meth_pred(name)
            }

            self.cls_registry[c.__name__] = RegisteredClass(
                cls=c,
                visibility=visibility,
                constructable=constructable,
                allowed_attrs=final_attrs,
                allowed_methods=final_methods,
            )
            return c

        return decorator(_cls) if _cls else decorator

    def module(
        self,
        mod: ModuleType,
        *,
        as_name: str,
        visibility: Visibility = "high",
        fns: Selector | None = Select.non_private,
        consts: Selector | None = Select.non_private,
        classes: Selector | None = Select.non_private,
        class_attrs: Selector | None = Select.non_private,
        class_methods: Selector | None = Select.non_private,
    ):
        """
        Registers a module's members under a namespace.
        """
        fn_pred = _create_predicate(fns)
        const_pred = _create_predicate(consts)
        class_pred = _create_predicate(classes)
        class_attr_pred = _create_predicate(class_attrs)
        class_meth_pred = _create_predicate(class_methods)

        selected_fns = set()
        selected_consts = set()
        selected_classes = {}

        for name, member in inspect.getmembers(mod):
            if inspect.isroutine(member):
                if fn_pred(name):
                    selected_fns.add(name)
            elif inspect.isclass(member):
                if class_pred(name):
                    # For each selected class, determine its own attrs and methods
                    attrs = {
                        m_name
                        for m_name, _ in inspect.getmembers(member)
                        if not callable(_) and class_attr_pred(m_name)
                    }
                    methods = {
                        m_name
                        for m_name, _ in inspect.getmembers(member)
                        if callable(_) and class_meth_pred(m_name)
                    }
                    selected_classes[name] = RegisteredClass(
                        cls=member,
                        visibility=visibility,
                        constructable=True,  # Default for module-registered classes
                        allowed_attrs=attrs,
                        allowed_methods=methods,
                    )
            elif (
                not callable(member)
                and not inspect.ismodule(member)
                and const_pred(name)
            ):
                selected_consts.add(name)

        self.module_registry[as_name] = RegisteredModule(
            module=mod,
            as_name=as_name,
            visibility=visibility,
            fns=selected_fns,
            consts=selected_consts,
            classes=selected_classes,
        )

    def task(self, func):
        pass
