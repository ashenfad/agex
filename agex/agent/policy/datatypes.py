from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Iterable, Literal, Union

from ..datatypes import MemberSpec

# Standalone datatypes for the security prototype (do not import core datatypes)

Pattern = Union[str, Iterable[str], Callable[[str], bool]]
Visibility = Literal["high", "medium", "low"]
RESERVED_NAMES = {"dataclass", "dataclasses"}


@dataclass
class ResolvedFn:
    fn: Callable


@dataclass
class ResolvedClass:
    cls: type
    constructable: bool


@dataclass
class ResolvedObj:
    value: Any


@dataclass
class Namespace:
    """
    Minimal namespace specification for lazy, unified registration.
    """

    name: str
    kind: Literal["module", "instance", "inherited", "virtual"]
    module: ModuleType | str | None = None
    # For instance namespaces, hold the live object
    obj: Any | None = None

    visibility: Visibility = "medium"
    include: Pattern | None = "*"
    exclude: Pattern | None = ("_*", "*._*")
    configure: dict[str, MemberSpec] = field(default_factory=dict)
    recursive: bool = False

    # For instance namespaces, support exception mappings (external -> agex)
    exception_mappings: dict[type, type] = field(default_factory=dict)

    # Inheritance: view onto another namespace
    parent: "Namespace | None" = None

    # Internal spec for virtual/main
    fns: dict[str, MemberSpec] = field(default_factory=dict, init=False)
    fn_objects: dict[str, Callable] = field(default_factory=dict, init=False)
    consts: dict[str, MemberSpec] = field(default_factory=dict, init=False)
    classes: dict[str, ResolvedClass] = field(default_factory=dict, init=False)

    def _ensure_module_loaded(self) -> ModuleType:
        if isinstance(self.module, ModuleType):
            return self.module
        if isinstance(self.module, str):
            mod = importlib.import_module(self.module)
            self.module = mod
            return mod
        raise TypeError("Namespace requires a module for module kind")


@dataclass
class ResolutionScope:
    # Agent's namespaces by agent-visible name (e.g., "__main__", "pandas")
    namespaces: dict[str, Namespace]
    # Index of real module path (e.g., "pandas", "pandas.core") -> list of Namespaces
    module_index: dict[str, list[Namespace]] = field(default_factory=dict)
    # Map object id(value) -> Namespace for registered live instances (future use)
    instance_map: dict[int, Namespace] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.module_index:
            index: dict[str, list[Namespace]] = {}
            for ns in self.namespaces.values():
                if ns.kind != "module":
                    continue
                # Prefer string module paths to avoid importing
                if isinstance(ns.module, str):
                    mpath = ns.module
                elif isinstance(ns.module, ModuleType):
                    mpath = ns.module.__name__
                else:
                    continue
                index.setdefault(mpath, []).append(ns)
                # Also index the root prefix to speed up prefix lookups
                root = mpath.split(".")[0]
                index.setdefault(root, []).append(ns)
            self.module_index = index
