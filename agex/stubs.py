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

    def needs_input(self, question) -> None:
        """Stop the task to ask whoever gave it something only they can
        settle; their answer starts the next run."""
        self._remote.needs_input(str(question))
        raise Finished("task.needs_input")


# -- agent-defined tasks -----------------------------------------------------------


class TaskFailed(Exception):
    """An agent-defined task handed back no value: its helper failed, or
    the value it made doesn't fit the task's return type."""


class TaskNeedsInput(TaskFailed):
    """An agent-defined task's helper asked a question (``question``)
    that only its caller can settle. No model is there to answer it, so
    the call fails with it: the code that called reads the question, and
    can call again with what it says."""

    def __init__(self, name, question) -> None:
        super().__init__(f"task {name!r} needs input: {question}")
        self.question = question


RESERVED_INPUTS = frozenset({"task", "host", "agex"})
"""Names an agent-defined task's parameters can't take: what its
helper's world binds."""

_FORMAT = "agex-task/1"


def _refuse_body(fn, name) -> None:
    """A task's body is its docstring; code there would never run."""
    import ast
    import inspect
    import textwrap

    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, SyntaxError):
        return
    node = tree.body[0] if tree.body else None
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        body = body[1:]
    for statement in body:
        if isinstance(statement, ast.Pass) or (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and statement.value.value is Ellipsis
        ):
            continue
        raise TypeError(
            f"task {name!r} has code in its body, which would never run: the "
            "docstring says what to do, and a helper agent does it. Leave the "
            "body empty (just the docstring)"
        )


def _compile(fn, names, held):
    """``fn`` as an agent-defined task: its name, its docstring and its
    parameters and return compiled (``nontainer.values``), refused with
    a ``TypeError`` saying what to write instead."""
    import inspect
    import typing

    from nontainer import values

    name = getattr(fn, "__name__", None)
    if not callable(fn) or not isinstance(name, str):
        raise TypeError("@agex.task decorates a function: def name(...) -> T")
    if name == "<lambda>":
        raise TypeError(
            "@agex.task needs a def, not a lambda: the def's name, docstring "
            "and annotations are what the task is. Write\n"
            "    @agex.task\n"
            "    def name(arg: T) -> R:\n"
            '        """What to do."""'
        )
    if inspect.iscoroutinefunction(fn):
        raise TypeError(
            f"task {name!r} is an async def; an agent-defined task is a plain "
            "def, and its call waits for the helper"
        )
    doc = inspect.getdoc(fn)
    if not doc:
        raise TypeError(
            f"task {name!r} has no docstring: the docstring is the job a helper "
            "agent does, so say what to do there"
        )
    _refuse_body(fn, name)
    signature = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn, localns=dict(names or {}))
    except NameError as error:
        raise TypeError(
            f"task {name!r} names a type that can't be resolved ({error}); "
            "define the types its annotations name before the task"
        ) from None
    params = []
    for param in signature.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError(
                f"task {name!r} takes *{param.name}: a task's inputs are named, "
                "each with its own annotation"
            )
        if param.name not in hints:
            raise TypeError(
                f"task {name!r}'s parameter {param.name!r} has no annotation: "
                "annotate each input with its type"
            )
        if param.name in RESERVED_INPUTS or param.name in held:
            taken = (
                "a name the helper's world binds itself"
                if param.name in RESERVED_INPUTS
                else "a host object of this world, which the helper's world holds too"
            )
            raise TypeError(
                f"task {name!r} has a parameter named {param.name!r}, {taken}; "
                "rename the parameter"
            )
        params.append(param.name)
    if "return" not in hints:
        raise TypeError(
            f"task {name!r} has no return annotation: annotate what it returns "
            "(-> None for nothing)"
        )
    specs = {}
    for key in (*params, "return"):
        try:
            spec = values.Spec.of(hints[key])
        except values.Unsupported as error:
            raise TypeError(f"task {name!r}: {error}") from None
        if not spec.travels:
            what = "returns" if key == "return" else f"takes {key!r} as"
            raise TypeError(
                f"task {name!r} {what} {values.fmt(hints[key])}, which has a live "
                "part: an agent-defined task takes and returns values (data, "
                "records, enums, bytes, tables, arrays)"
            )
        specs[key] = spec
    try:
        exported = values.export_specs(specs)
    except values.Unsupported as error:
        raise TypeError(f"task {name!r}: {error}") from None
    data = {
        "format": _FORMAT,
        "name": name,
        "doc": doc,
        "params": params,
        "specs": exported,
    }
    return data, signature, specs


class AgentTask:
    """A task agent code defined with ``@agex.task``: call it for its
    value, built as the return annotation says, or ``.map`` it over many
    inputs at once."""

    def __init__(self, remote, fn, primer, names, held) -> None:
        self._remote = remote
        self._primer = primer or ""
        self._data, self._signature, self._specs = _compile(fn, names, held)
        self.__name__ = self._data["name"]
        self.__doc__ = self._data["doc"]
        self.__wrapped__ = fn

    def __repr__(self) -> str:
        return f"<agex task {self.__name__}{self._signature}>"

    def _inputs(self, args, kwargs):
        from nontainer import values

        name = self.__name__
        try:
            bound = self._signature.bind(*args, **kwargs)
        except TypeError as error:
            raise TypeError(f"{name}(): {error}") from None
        bound.apply_defaults()
        blobs = {}
        for key, value in bound.arguments.items():
            spec = self._specs[key]
            try:
                spec.check(value)
            except values.Mismatch as mismatch:
                raise TypeError(
                    f"{name}()'s argument {key!r} must be {values.fmt(spec.annotation)}: "
                    f"{mismatch}"
                ) from None
            blobs[key] = values.encode(value).to_bytes()
        return blobs

    def _value(self, reply):
        from nontainer import values

        name = self.__name__
        status = reply.get("status")
        if status == "needs_input":
            raise TaskNeedsInput(name, reply.get("message") or "")
        if status != "success":
            raise TaskFailed(f"task {name!r} failed: {reply.get('message') or status}")
        returns = self._specs["return"]
        try:
            return returns.decode(reply.get("value") or b"")
        except Exception as error:  # noqa: BLE001 - a class's own check, say
            raise TaskFailed(
                f"task {name!r} handed back a value that isn't "
                f"{values.fmt(returns.annotation)}: {str(error) or type(error).__name__}"
            ) from error

    def __call__(self, *args, **kwargs):
        reply = self._remote.call(self._data, self._inputs(args, kwargs), self._primer)
        return self._value(reply)

    def map(self, items):
        """Call the task once per item, at once (up to the limit the world
        sets), and return the values in order. With one parameter each
        item is its argument; with more, a tuple is the positional
        arguments and a dict the keyword ones. The first call that fails
        raises, once all of them have finished."""
        calls = []
        for item in items:
            if len(self._specs) == 2:
                calls.append(self._inputs((item,), {}))
            elif isinstance(item, dict):
                calls.append(self._inputs((), item))
            elif isinstance(item, tuple):
                calls.append(self._inputs(item, {}))
            else:
                raise TypeError(
                    f"{self.__name__}.map(): with more than one parameter, each "
                    "item is a tuple of arguments or a dict of them"
                )
        replies = self._remote.map(self._data, calls, self._primer)
        return [self._value(reply) for reply in replies]


class AgexStub:
    """``agex`` in agent code: define a task, a function a helper agent
    runs, typed by its annotations.

        @agex.task                      # or @agex.task(primer="You do geometry.")
        def corners(shape: str) -> list[Point2D]:
            \"""The corner points of the named shape.\"""

        pts = corners("unit square")    # your own Point2D objects
        many = corners.map(["square", "hexagon"])

    The docstring is the job, and the body stays empty. Inputs and the
    result are values (data, records and enums of your own, bytes,
    tables, arrays), checked against the annotations. A helper that
    can't do it raises ``agex.TaskFailed``; one that asks a question
    raises ``agex.TaskNeedsInput`` with it.
    """

    TaskFailed = TaskFailed
    TaskNeedsInput = TaskNeedsInput

    def __init__(self, remote) -> None:
        self._remote = remote

    def __repr__(self) -> str:
        return "<agex: @agex.task defines a task>"

    def task(self, fn=None, *, primer=None):
        """Make ``fn`` a task. ``primer`` is added to what the helper
        agent is told about how to work."""
        import sys

        names = dict(sys._getframe(1).f_locals)

        def make(fn):
            return AgentTask(self._remote, fn, primer, names, set(self._remote.held()))

        return make if fn is None else make(fn)
