"""Policy-backed lazy registration and resolution API."""

from ..datatypes import MemberSpec
from .datatypes import (
    Namespace,
    Pattern,
    ResolutionScope,
    ResolvedClass,
    ResolvedFn,
    ResolvedObj,
    Visibility,
)
from .policy import AgentPolicy
from .resolve import make_predicate

__all__ = [
    "AgentPolicy",
    "Namespace",
    "Pattern",
    "ResolutionScope",
    "ResolvedClass",
    "ResolvedFn",
    "ResolvedObj",
    "Visibility",
    "MemberSpec",
    "make_predicate",
]
