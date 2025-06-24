import pickle
from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping


class KVStore(ABC):
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        pass

    @abstractmethod
    def get_many(self, *args: str) -> Mapping[str, Any]:
        pass

    @abstractmethod
    def set_many(self, **kwargs) -> None:
        pass

    @abstractmethod
    def items(self) -> Iterable[tuple[str, Any]]:
        pass

    @abstractmethod
    def __contains__(self, key: str) -> bool:
        pass


class Memory(KVStore):
    def __init__(self, as_bytes: bool = True):
        """
        A memory-backed KV store. If `as_bytes` is True, values are stored as
        bytes, effectively making the values immutable.
        """
        self.memory = {}
        self.as_bytes = as_bytes

    def get(self, key: str, default: Any = None) -> Any:
        value = self.memory.get(key)
        if self.as_bytes and value is not None:
            value = pickle.loads(value)
        return value if value is not None else default

    def set(self, key: str, value: Any) -> None:
        self.memory[key] = pickle.dumps(value) if self.as_bytes else value

    def get_many(self, *args: str) -> Mapping[str, Any]:
        return {
            key: pickle.loads(val) if self.as_bytes else val
            for key in args
            if (val := self.memory.get(key)) is not None
        }

    def set_many(self, **kwargs) -> None:
        if self.as_bytes:
            coll = {key: pickle.dumps(value) for key, value in kwargs.items()}
            self.memory.update(coll)
        else:
            self.memory.update(kwargs)

    def items(self) -> Iterable[tuple[str, Any]]:
        return self.memory.items()

    def __contains__(self, key: str) -> bool:
        return key in self.memory
