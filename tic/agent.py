from dataclasses import dataclass
from typing import Callable


@dataclass
class RegisteredItem:
    public: bool  # expose existence to agent
    document: bool  # expose documentation to agent


@dataclass
class RegisteredFn(RegisteredItem):
    fn: Callable


@dataclass
class RegisteredClass(RegisteredItem):
    cls: type
    constructable: bool
    allowed_methods: set[str]  # perhaps make this a set of RegisteredFn?
    allowed_attrs: set[str]


class Agent:
    def __init__(self, primer: str | None = None):
        self.primer = primer
        self.fn_registry: dict[str, RegisteredFn] = {}
        self.cls_registry: dict[str, RegisteredClass] = {}
        self.fn = self.make_fn_decorator()
        self.cls = self.make_cls_decorator()
        self.task = self.make_task_decorator()

    def make_fn_decorator(self):
        pass

    def make_cls_decorator(self):
        pass

    def make_task_decorator(self):
        pass
