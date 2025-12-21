"""
Loop package for agent task execution.

This package provides the TaskLoopMixin that handles the core think→act loop
for agent tasks, including LLM communication and code evaluation.
"""

from .mixin import TaskLoopMixin

__all__ = ["TaskLoopMixin"]
