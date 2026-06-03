"""Tests for ``spawn`` — ephemeral in-agent clones (agex/agent/spawn.py).

The agent defines ``@spawn.task`` functions and runs them as memoryless clones
of itself, either directly (blocking), via ``spawn.submit`` (Future), or
``spawn.map``. Clones share the parent's policy on fresh ephemeral state.

Concurrency tests use a thread-safe Dummy subclass because the stock ``Dummy``
serves canned responses via a non-locked counter (real LLM calls are
independent). Clone responses are kept identical where order across racing
threads would otherwise matter.
"""

import threading

import pytest

from agex import Agent, clear_agent_registry
from agex.llm.dummy_client import Dummy
from tests.agex._emissions import make_response


class SafeDummy(Dummy):
    """Thread-safe Dummy: serializes the response counter for concurrent clones."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()

    def complete(self, system, events, **kwargs):
        with self._lock:
            return super().complete(system, events, **kwargs)


@pytest.fixture(autouse=True)
def _clear():
    clear_agent_registry()


def _clone(value: str) -> object:
    """A clone turn that returns ``value`` via task_success."""
    return make_response(thinking="clone", code=f"task_success({value})")


# --------------------------------------------------------------------------- #
# Surface: direct call / submit / map / heterogeneous                         #
# --------------------------------------------------------------------------- #


def test_direct_call_returns_typed_result():
    responses = [
        make_response(
            thinking="spawn",
            code=(
                "@spawn.task\n"
                "def gen(x: int) -> int:\n"
                '    """double"""\n'
                "    pass\n"
                "task_success(gen(21))\n"
            ),
        ),
        _clone("42"),
    ]
    parent = Agent(name="p_direct", llm=Dummy(responses=responses))

    @parent.task
    def run() -> int:
        """use spawn"""
        pass

    assert run() == 42


def test_submit_result():
    responses = [
        make_response(
            thinking="submit",
            code=(
                "@spawn.task\n"
                "def gen(x: int) -> int:\n"
                '    """double"""\n'
                "    pass\n"
                "h = spawn.submit(gen, 21)\n"
                "task_success(h.result())\n"
            ),
        ),
        _clone("42"),
    ]
    parent = Agent(name="p_submit", llm=Dummy(responses=responses))

    @parent.task
    def run() -> int:
        """use spawn"""
        pass

    assert run() == 42


def test_map_concurrent():
    responses = [
        make_response(
            thinking="map",
            code=(
                "@spawn.task\n"
                "def gen(x: int) -> int:\n"
                '    """double"""\n'
                "    pass\n"
                "task_success(spawn.map(gen, [1, 2, 3]))\n"
            ),
        ),
        _clone("10"),
        _clone("10"),
        _clone("10"),
    ]
    parent = Agent(name="p_map", llm=SafeDummy(responses=responses), max_spawns=3)

    @parent.task
    def run() -> list:
        """use spawn"""
        pass

    assert run() == [10, 10, 10]


def test_heterogeneous_fanout():
    responses = [
        make_response(
            thinking="hetero",
            code=(
                "@spawn.task\n"
                "def gen_a(x: int) -> int:\n"
                '    """a"""\n'
                "    pass\n"
                "@spawn.task\n"
                "def gen_b(s: str) -> str:\n"
                '    """b"""\n'
                "    pass\n"
                "ha = spawn.submit(gen_a, 9)\n"
                "hb = spawn.submit(gen_b, 'x')\n"
                "task_success({'a': ha.result(), 'b': hb.result()})\n"
            ),
        ),
        _clone("7"),
        _clone("'B!'"),
    ]
    parent = Agent(name="p_hetero", llm=SafeDummy(responses=responses), max_spawns=2)

    @parent.task
    def run() -> dict:
        """use spawn"""
        pass

    # Identical-shaped responses aside, the two clones return distinct typed
    # values; order across two submits is deterministic enough with a lock, but
    # assert membership to avoid coupling to scheduling.
    out = run()
    assert set(out.keys()) == {"a", "b"}
    assert out["a"] == 7
    assert out["b"] == "B!"


# --------------------------------------------------------------------------- #
# Future API surface reachable from sandboxed code                            #
# --------------------------------------------------------------------------- #


def test_future_api_in_sandbox():
    responses = [
        make_response(
            thinking="future-api",
            code=(
                "@spawn.task\n"
                "def gen(x: int) -> int:\n"
                '    """d"""\n'
                "    pass\n"
                "h = spawn.submit(gen, 1)\n"
                "v = h.result()\n"
                "task_success((v, h.done(), h.exception() is None))\n"
            ),
        ),
        _clone("7"),
    ]
    parent = Agent(name="p_future", llm=Dummy(responses=responses))

    @parent.task
    def run() -> tuple:
        """use spawn"""
        pass

    assert run() == (7, True, True)


# --------------------------------------------------------------------------- #
# Isolation: clone is depth-1, blank, and doesn't mutate the parent           #
# --------------------------------------------------------------------------- #


def test_clone_is_depth_1_and_isolated():
    import agex
    from agex.eval.bridge.namespace import build_namespace

    parent = Agent(name="p_iso")
    parent.module(agex)
    before = set(parent._policy.namespaces.keys())

    clone = parent._get_spawn_clone()
    assert clone._spawn_enabled is False
    assert clone._state_config is None  # ephemeral

    # Depth-1: the clone's own namespace has no `spawn`.
    clone_ns, _ = build_namespace(state=None, agent=clone, agent_name=clone.name)
    assert "spawn" not in clone_ns

    # Parent's namespace does have spawn.
    parent_ns, _ = build_namespace(state=None, agent=parent, agent_name=parent.name)
    assert "spawn" in parent_ns

    # Registering a task on the clone must not leak into the parent policy.
    @clone.task
    def child(x: int) -> int:
        """child"""
        pass

    assert set(parent._policy.namespaces.keys()) == before
    assert "child" not in parent._tasks


def test_spawn_clone_is_cached():
    parent = Agent(name="p_cache")
    assert parent._get_spawn_clone() is parent._get_spawn_clone()


# --------------------------------------------------------------------------- #
# Observability: clone events forward, labeled by namespace                   #
# --------------------------------------------------------------------------- #


def test_clone_events_are_namespaced():
    responses = [
        make_response(
            thinking="label",
            code=(
                "@spawn.task\n"
                "def gen(x: int) -> int:\n"
                '    """d"""\n'
                "    pass\n"
                "task_success(gen(1))\n"
            ),
        ),
        _clone("5"),
    ]
    parent = Agent(name="p_label", llm=Dummy(responses=responses))

    @parent.task
    def run() -> int:
        """use spawn"""
        pass

    seen = []
    result = run(on_event=lambda e: seen.append(getattr(e, "full_namespace", None)))
    assert result == 5
    # Clone events labeled by the per-invocation tag; parent events by name.
    assert "spawn:gen:0" in seen
    assert "p_label" in seen


# --------------------------------------------------------------------------- #
# Errors: a clone failure is recoverable for the parent agent                 #
# --------------------------------------------------------------------------- #


def test_clone_failure_is_recoverable():
    responses = [
        make_response(
            thinking="catch",
            code=(
                "@spawn.task\n"
                "def gen(x: int) -> int:\n"
                '    """d"""\n'
                "    pass\n"
                "try:\n"
                "    gen(1)\n"
                "    task_success('no-error')\n"
                "except Exception:\n"
                "    task_success('caught')\n"
            ),
        ),
        make_response(thinking="boom", code="task_fail('nope')"),
    ]
    parent = Agent(name="p_err", llm=Dummy(responses=responses))

    @parent.task
    def run() -> str:
        """use spawn"""
        pass

    assert run() == "caught"


# --------------------------------------------------------------------------- #
# Concurrency: pool bounded by max_spawns, torn down per emission             #
# --------------------------------------------------------------------------- #


def test_pool_bounded_and_torn_down():
    responses = [
        make_response(
            thinking="fanout",
            code=(
                "@spawn.task\n"
                "def gen(x: int) -> int:\n"
                '    """d"""\n'
                "    pass\n"
                "hs = [spawn.submit(gen, i) for i in range(6)]\n"
                "task_success(sum(h.result() for h in hs))\n"
            ),
        ),
    ] + [_clone("10") for _ in range(6)]
    parent = Agent(name="p_pool", llm=SafeDummy(responses=responses), max_spawns=2)

    @parent.task
    def run() -> int:
        """use spawn"""
        pass

    assert run() == 60  # 6 clones x 10, queued through a 2-worker pool
    leftover = [
        t.name for t in threading.enumerate() if t.name.startswith("agex-spawn")
    ]
    assert leftover == []


def test_max_spawns_param():
    assert Agent(name="ms3", max_spawns=3).max_spawns == 3
    assert Agent(name="ms_default").max_spawns == 8


# --------------------------------------------------------------------------- #
# Async parent: spawn works under an async task (threads, not the loop)       #
# --------------------------------------------------------------------------- #


def test_spawn_under_async_parent():
    import asyncio

    responses = [
        make_response(
            thinking="async-parent",
            code=(
                "@spawn.task\n"
                "def gen(x: int) -> int:\n"
                '    """d"""\n'
                "    pass\n"
                "task_success(gen(21))\n"
            ),
        ),
        _clone("42"),
    ]
    parent = Agent(name="p_async", llm=Dummy(responses=responses))

    @parent.task
    async def run() -> int:
        """use spawn"""
        pass

    assert asyncio.run(run()) == 42


# --------------------------------------------------------------------------- #
# Decorator-factory form, bad-arg rejection, scope-request surfacing          #
# --------------------------------------------------------------------------- #


def test_spawn_task_factory_form():
    """`@spawn.task()` (called, possibly with kwargs) works like the bare form
    — the forward-compat guardrail so `@spawn.task(fs=...)` can be added later."""
    responses = [
        make_response(
            thinking="factory",
            code=(
                "@spawn.task()\n"
                "def gen(x: int) -> int:\n"
                '    """double"""\n'
                "    pass\n"
                "task_success(gen(21))\n"
            ),
        ),
        _clone("42"),
    ]
    parent = Agent(name="p_factory", llm=Dummy(responses=responses))

    @parent.task
    def run() -> int:
        """use spawn"""
        pass

    assert run() == 42


def test_submit_rejects_non_task():
    """submit/map raise TypeError when the first arg isn't a @spawn.task."""
    from agex.agent.spawn import Spawn

    parent = Agent(name="p_badarg")
    spawn = Spawn(parent._get_spawn_clone(), parent)
    with pytest.raises(TypeError):
        spawn.submit(object(), 1)
    with pytest.raises(TypeError):
        spawn.map(lambda y: y, [1])


def test_clone_scope_request_is_recoverable():
    """A clone that requests a capability scope cannot durably suspend (it's
    ephemeral); the need surfaces to the parent as a recoverable EvalError
    naming the scope, raised through the Future."""
    from agex.agent.spawn import Spawn
    from agex.eval.error import EvalError

    llm = Dummy(
        responses=[
            make_response(thinking="ask", code="task_request_permission('email')")
        ]
    )
    parent = Agent(name="p_scope", llm=llm)
    spawn = Spawn(parent._get_spawn_clone(), parent)

    def gen(x: int) -> int:
        """d"""
        pass

    gen_task = spawn.task(gen)
    handle = spawn.submit(gen_task, 1)  # host-side submit: exercises the
    with pytest.raises(EvalError) as excinfo:  # raw-wrapper resolve path too
        handle.result()
    assert "email" in str(excinfo.value)
    spawn.close()


# --------------------------------------------------------------------------- #
# Sandbox-defined types flow into clones (roadmap/spawn-type-sharing.md)       #
# --------------------------------------------------------------------------- #


def test_sandbox_class_return_type():
    """An agent can define a class in its own sandbox and use it as a
    @spawn.task return type — the class is seeded into the clone, which
    constructs it; the instance crosses back with attributes intact."""
    responses = [
        make_response(
            thinking="define+spawn",
            code=(
                "class Tile:\n"
                "    def __init__(self, name):\n"
                "        self.name = name\n"
                "@spawn.task\n"
                "def make_tile(p: str) -> Tile:\n"
                '    """Make a tile named after the prompt."""\n'
                "    pass\n"
                'task_success(make_tile("castle").name)\n'
            ),
        ),
        make_response(thinking="clone", code='task_success(Tile(name="castle"))'),
    ]
    parent = Agent(name="p_sbtype", llm=Dummy(responses=responses))

    @parent.task
    def run() -> str:
        """t"""
        pass

    assert run() == "castle"


def test_sandbox_class_return_is_validated():
    """The seeded class is the real expected type: a clone that returns the
    wrong type is rejected and recovers (the contract is enforced, not a
    Pydantic no-op)."""
    responses = [
        make_response(
            thinking="define+spawn",
            code=(
                "class Tile:\n"
                "    def __init__(self, name):\n"
                "        self.name = name\n"
                "@spawn.task\n"
                "def make_tile(p: str) -> Tile:\n"
                '    """make a tile"""\n'
                "    pass\n"
                'task_success(make_tile("castle").name)\n'
            ),
        ),
        # clone returns the wrong type first (must be rejected), then corrects
        make_response(thinking="oops", code='task_success("just a string")'),
        make_response(thinking="fixed", code='task_success(Tile(name="castle"))'),
    ]
    parent = Agent(name="p_sbvalid", llm=Dummy(responses=responses))

    @parent.task
    def run() -> str:
        """t"""
        pass

    assert run() == "castle"
    # parent turn + two clone turns (wrong, then corrected)
    assert parent.llm.call_count == 3


def test_sandbox_class_return_type_under_fanout():
    """A sandbox-defined return type also works through the threaded
    submit/map path: each clone reconstructs a fresh copy of the class
    (per-invocation), so concurrent fan-out is safe."""
    responses = [
        make_response(
            thinking="fanout",
            code=(
                "class Tile:\n"
                "    def __init__(self, name):\n"
                "        self.name = name\n"
                "@spawn.task\n"
                "def make_tile(p: str) -> Tile:\n"
                '    """make a tile"""\n'
                "    pass\n"
                'tiles = spawn.map(make_tile, ["a", "b", "c"])\n'
                "task_success([t.name for t in tiles])\n"
            ),
        ),
    ] + [
        make_response(thinking="c", code='task_success(Tile(name="x"))')
        for _ in range(3)
    ]
    parent = Agent(name="p_sbfanout", llm=SafeDummy(responses=responses), max_spawns=3)

    @parent.task
    def run() -> list:
        """t"""
        pass

    assert run() == ["x", "x", "x"]


def test_sandbox_class_param():
    """A sandbox-defined class works as a @spawn.task *parameter*: the parent
    constructs the instance, it crosses in, and the clone reads it (duck-typed).
    Input validation passes via the same StInstance-identity check."""
    responses = [
        make_response(
            thinking="define+spawn",
            code=(
                "class Tile:\n"
                "    def __init__(self, name):\n"
                "        self.name = name\n"
                "@spawn.task\n"
                "def describe(tile: Tile) -> str:\n"
                '    """describe the tile"""\n'
                "    pass\n"
                'task_success(describe(Tile(name="castle")))\n'
            ),
        ),
        make_response(
            thinking="clone",
            code='task_success("a tile called " + inputs.tile.name)',
        ),
    ]
    parent = Agent(name="p_sbparam", llm=Dummy(responses=responses))

    @parent.task
    def run() -> str:
        """t"""
        pass

    assert run() == "a tile called castle"


def test_generic_sandbox_return_fails_fast():
    """A sandbox class nested in a generic return type isn't supported yet; it
    must fail fast at definition with a clear, catchable error — NOT spin to a
    TaskTimeout."""
    responses = [
        make_response(
            thinking="x",
            code=(
                "class Tile:\n"
                "    def __init__(self, name):\n"
                "        self.name = name\n"
                "try:\n"
                "    @spawn.task\n"
                "    def make_tiles(n: int) -> list[Tile]:\n"
                '        """make n tiles"""\n'
                "        pass\n"
                '    task_success("NO ERROR")\n'
                "except Exception as e:\n"
                '    task_success("caught" if "nested in a generic" in str(e) else "miss")\n'
            ),
        ),
    ]
    parent = Agent(name="p_genfail", llm=Dummy(responses=responses))

    @parent.task
    def run() -> str:
        """t"""
        pass

    assert run() == "caught"
    # one turn only — no spin/timeout
    assert parent.llm.call_count == 1
