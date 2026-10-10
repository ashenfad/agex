"""agex under the conformance corpora.

- **The harness corpus** (nontainer's): :class:`AgexHarness` runs agex's
  loop against the contract the agno adapter is held to.
- **The shape corpus** (agex's, in :mod:`agex.conformance.scenarios`):
  :class:`AgexTasks` runs agex's task API, the way agex-ts will be run.
  Its format is :mod:`agex.conformance.shape`, its runner
  :mod:`agex.conformance.runner`, and ``python -m
  agex.conformance.export`` writes it as JSON for other languages. Its
  code scenarios (:mod:`agex.conformance.code_scenarios`) pin the tasks
  agent code defines, and :class:`~agex.conformance.code_tasks.AgexCodeTasks`
  runs them.
"""

from .harness import AgexHarness, AgexSession
from .tasks import AgexTasks

__all__ = ["AgexHarness", "AgexSession", "AgexTasks"]
