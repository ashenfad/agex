"""What agent code holds in a task's world: the stubs in front of agex's
host objects (``nontainer.HostObject(..., stub=...)``).

A stub runs where the agent's code runs, in this process, a worker
process or a dud machine, and reaches its host object through
``remote``, whose calls nontainer types by the host object's
annotations. Standard library only, importing nothing from agex: a dud
guest rebuilds this module from its source.
"""


class Finished(BaseException):
    """Stops the script that ended its task. A ``BaseException``, so the
    agent code's ``except Exception`` lets it through."""


class TaskStub:
    """``task`` in agent code: how a task's run hands back its result.

    Each call reaches the task's host half, which records the result,
    then stops the script. A value that doesn't fit the task's return
    type is a ``TypeError`` raised at the call, before anything is
    recorded, so the code can fix it and call again.
    """

    def __init__(self, remote) -> None:
        self._remote = remote

    def __repr__(self) -> str:
        return "<task>"

    def success(self, value=None) -> None:
        """End the task with ``value``, which must fit its return type."""
        self._remote.success(value)
        raise Finished("task.success")

    def fail(self, reason) -> None:
        """End the task without a value, saying why it can't be done."""
        self._remote.fail(str(reason))
        raise Finished("task.fail")
