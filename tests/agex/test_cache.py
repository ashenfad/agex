"""Tests for the agent-session-scoped cache primitive."""

import pickle

import pytest

from agex import Agent, clear_agent_registry, connect_state
from agex.agent.datatypes import TaskFail, TaskSuccess
from agex.cache import PREFIX, Cache
from agex.eval.bridge import execute_sandboxed
from agex.llm import Dummy
from agex.state.live import Live
from tests.agex._emissions import make_response

# -----------------------------------------------------------------------------
# Cache class — direct tests
# -----------------------------------------------------------------------------


class TestCacheDirect:
    """Unit tests for the Cache wrapper itself."""

    def test_set_get(self):
        state: dict = {}
        cache = Cache(state)
        cache["foo"] = 42
        assert cache["foo"] == 42
        assert state[PREFIX + "foo"] == 42

    def test_default_get(self):
        cache = Cache({})
        assert cache.get("missing") is None
        assert cache.get("missing", "default") == "default"

    def test_contains(self):
        state: dict = {}
        cache = Cache(state)
        cache["foo"] = 1
        assert "foo" in cache
        assert "bar" not in cache
        assert "__cache__/foo" not in cache  # qualified key is not a user key
        assert 42 not in cache  # non-string

    def test_delitem(self):
        cache = Cache({})
        cache["foo"] = 1
        del cache["foo"]
        assert "foo" not in cache
        with pytest.raises(KeyError):
            del cache["missing"]

    def test_iter_strips_prefix(self):
        state: dict = {"unrelated": 1, PREFIX + "a": 1, PREFIX + "b": 2}
        cache = Cache(state)
        assert sorted(cache) == ["a", "b"]

    def test_len_counts_only_cache_keys(self):
        state: dict = {"unrelated": 1, PREFIX + "a": 1, PREFIX + "b": 2}
        cache = Cache(state)
        assert len(cache) == 2

    def test_does_not_leak_other_state_keys(self):
        """Cache iteration / membership ignores non-cache state keys."""
        state: dict = {"inputs": "hello", "__event_log__": [], "_event_1": "x"}
        cache = Cache(state)
        assert list(cache) == []
        assert len(cache) == 0
        assert "inputs" not in cache
        assert "__event_log__" not in cache

    def test_reject_dunder_keys(self):
        cache = Cache({})
        with pytest.raises(ValueError, match="reserved for framework"):
            cache["__foo"] = 1

    def test_reject_slash_keys(self):
        cache = Cache({})
        with pytest.raises(ValueError, match="reserved for namespacing"):
            cache["a/b"] = 1

    def test_reject_non_string_keys(self):
        cache = Cache({})
        with pytest.raises(TypeError):
            cache[42] = "x"

    def test_repr_lists_keys(self):
        cache = Cache({})
        cache["b"] = 2
        cache["a"] = 1
        assert repr(cache) == "Cache(['a', 'b'])"

    def test_dict_protocol_methods(self):
        """MutableMapping mixins (keys, values, items, update) work."""
        cache = Cache({})
        cache.update({"a": 1, "b": 2})
        assert dict(cache.items()) == {"a": 1, "b": 2}
        assert sorted(cache.keys()) == ["a", "b"]
        assert sorted(cache.values()) == [1, 2]


# -----------------------------------------------------------------------------
# Cache as injected namespace value — integration with execute_sandboxed
# -----------------------------------------------------------------------------


class TestCacheNamespaceInjection:
    """Cache appears in the agent's namespace and round-trips through state."""

    def setup_method(self):
        clear_agent_registry()

    def test_cache_is_present_in_namespace(self):
        agent = Agent(name="t")
        state = Live()
        state["__event_log__"] = []
        ns = execute_sandboxed("cache['probe'] = 1", agent, state)
        assert isinstance(ns["cache"], Cache)
        assert ns["cache"]["probe"] == 1

    def test_agent_writes_persist_in_state(self):
        agent = Agent(name="t")
        state = Live()
        state["__event_log__"] = []
        execute_sandboxed("cache['foo'] = 42", agent, state)
        # Cache values land under the prefix in state
        assert state.get(PREFIX + "foo") == 42

    def test_writes_in_one_action_visible_in_next(self):
        agent = Agent(name="t")
        state = Live()
        state["__event_log__"] = []
        execute_sandboxed("cache['x'] = 10", agent, state)
        # Variables don't carry between actions, but cache values do.
        with pytest.raises(TaskSuccess) as exc:
            execute_sandboxed("task_success(cache['x'] + 1)", agent, state)
        assert exc.value.result == 11

    def test_unpicklable_value_raises_to_agent(self):
        """Storing an unpicklable object in cache surfaces the error so
        the agent can choose a different representation."""
        import threading

        from agex.cache import CacheError

        agent = Agent(name="t")
        agent.fn(threading.Lock, name="make_lock")
        state = Live()
        state["__event_log__"] = []
        # threading.Lock instances are not picklable by cloudpickle.
        # The cache check raises at write time, not silently markers it.
        with pytest.raises(CacheError):
            execute_sandboxed("cache['lock'] = make_lock()", agent, state)

    def test_picklable_value_passes_validation(self):
        """The validator should accept normal picklable values."""
        agent = Agent(name="t")
        state = Live()
        state["__event_log__"] = []
        execute_sandboxed("cache['data'] = [1, 2, 3]", agent, state)
        assert state.get(PREFIX + "data") == [1, 2, 3]

    def test_lambda_is_acceptable(self):
        """cloudpickle handles lambdas; the validator should accept them."""
        agent = Agent(name="t")
        state = Live()
        state["__event_log__"] = []
        # lambdas are picklable via cloudpickle
        execute_sandboxed("cache['fn'] = lambda x: x + 1", agent, state)
        assert callable(state.get(PREFIX + "fn"))


# -----------------------------------------------------------------------------
# Cache survives across tasks within an agent session
# -----------------------------------------------------------------------------


class TestCacheAcrossTasks:
    """Cache values written in one task are visible in the next."""

    def setup_method(self):
        clear_agent_registry()

    def test_persistence_across_tasks(self):
        config = connect_state(type="versioned", storage="memory")
        agent = Agent(name="t", llm=Dummy(), state=config)

        @agent.task
        def stash() -> None:
            """Stash a value in the cache."""
            pass

        @agent.task
        def recall() -> int:
            """Recall the cached value."""
            pass

        agent.llm.responses = [
            make_response(
                thinking="store and finish",
                code='cache["answer"] = 42\ntask_success(None)',
            )
        ]
        stash(session="s")

        agent.llm.responses = [
            make_response(thinking="recall", code='task_success(cache["answer"])')
        ]
        result = recall(session="s")
        assert result == 42

    def test_separate_sessions_have_separate_caches(self):
        config = connect_state(type="versioned", storage="memory")
        agent = Agent(name="t", llm=Dummy(), state=config)

        @agent.task
        def stash() -> None:
            """Stash a value."""
            pass

        @agent.task
        def recall() -> int:
            """Recall a value."""
            pass

        agent.llm.responses = [
            make_response(
                thinking="store",
                code='cache["x"] = 100\ntask_success(None)',
            )
        ]
        stash(session="session_a")

        # Different session, same agent → cache is fresh.
        agent.llm.responses = [
            make_response(
                thinking="check",
                code='task_success(cache.get("x", -1))',
            )
        ]
        result = recall(session="session_b")
        assert result == -1


# -----------------------------------------------------------------------------
# Sub-agent isolation
# -----------------------------------------------------------------------------


class TestCacheSubAgentIsolation:
    """Each sub-agent has its own cache, isolated from parent and siblings."""

    def setup_method(self):
        clear_agent_registry()

    def test_sub_agent_cache_independent_of_parent(self):
        specialist = Agent(
            name="specialist",
            llm=Dummy(),
            state=connect_state(type="versioned", storage="memory"),
        )
        orchestrator = Agent(
            name="orchestrator",
            llm=Dummy(),
            state=connect_state(type="versioned", storage="memory"),
        )

        @orchestrator.fn
        @specialist.task
        def specialist_probe() -> int:
            """Return the specialist's view of cache['x'], or -1."""
            pass

        @orchestrator.task
        def main() -> tuple[int, int]:
            """Compare parent and child caches."""
            pass

        # Orchestrator stores in its own cache, then calls specialist.
        # Specialist stores a different value in its own cache.
        orchestrator.llm.responses = [
            make_response(
                thinking="orchestrate",
                code=(
                    'cache["x"] = 10\n'
                    "spec_x = specialist_probe()\n"
                    'task_success((cache["x"], spec_x))'
                ),
            )
        ]
        specialist.llm.responses = [
            make_response(
                thinking="specialist",
                code='cache["x"] = 99\ntask_success(cache["x"])',
            )
        ]
        result = main(session="s")
        assert result == (10, 99), (
            "parent cache should keep its value (10); sub-agent should see its own (99)"
        )


# -----------------------------------------------------------------------------
# Sandbox-defined function survives via cache
# -----------------------------------------------------------------------------


class TestCacheStFunction:
    """Sandbox-defined functions can be cached and called across tasks.

    Cross-task activation is delivered by sandtrap's
    ``__sandtrap_activate__`` container hook (>= 0.2.0), which Cache
    implements: on every ``exec`` sandtrap walks Cache values and
    re-activates any inactive ``StFunction``/``StClass`` it finds.
    """

    def setup_method(self):
        clear_agent_registry()

    def test_sandbox_function_storeable_within_task(self):
        """An StFunction can be cached and retrieved within the same
        task (where it remains active in memory)."""
        agent = Agent(name="t")
        state = Live()
        state["__event_log__"] = []
        execute_sandboxed(
            'def add(a, b):\n    return a + b\ncache["add"] = add',
            agent,
            state,
        )
        assert PREFIX + "add" in state

    def test_sandbox_function_round_trips_across_tasks(self):
        """A function cached in one task is callable in the next."""
        config = connect_state(type="versioned", storage="memory")
        agent = Agent(name="t", llm=Dummy(), state=config)

        @agent.task
        def define() -> None:
            """Define a helper and stash it."""
            pass

        @agent.task
        def use() -> int:
            """Retrieve and call the helper."""
            pass

        agent.llm.responses = [
            make_response(
                thinking="define and stash",
                code=(
                    "def add(a, b):\n    return a + b\n"
                    'cache["add"] = add\n'
                    "task_success(None)"
                ),
            )
        ]
        define(session="s")

        agent.llm.responses = [
            make_response(
                thinking="retrieve and use",
                code='fn = cache["add"]\ntask_success(fn(2, 3))',
            )
        ]
        result = use(session="s")
        assert result == 5


# -----------------------------------------------------------------------------
# Picklability of the Cache wrapper itself (cross-process isolation)
# -----------------------------------------------------------------------------


class TestCachePicklability:
    """The Cache wrapper survives pickle roundtrip for cross-process use."""

    def test_cache_is_picklable_when_state_is(self):
        state: dict = {PREFIX + "k": 1}
        cache = Cache(state)
        roundtripped = pickle.loads(pickle.dumps(cache))
        assert roundtripped["k"] == 1
        assert "k" in roundtripped


# -----------------------------------------------------------------------------
# Primer text mentions cache
# -----------------------------------------------------------------------------


def test_primer_mentions_cache():
    """The cache is documented in the agent's system message."""
    agent = Agent()
    system_message = agent._build_system_message()
    assert "cache" in system_message.lower()
    assert "cache[" in system_message  # syntax example


# -----------------------------------------------------------------------------
# Internal-key isolation
# -----------------------------------------------------------------------------


def test_cache_does_not_collide_with_internal_keys():
    """Cache writes never collide with framework-managed state keys."""
    state: dict = {
        "__event_log__": [],
        "__expected_return_type__": int,
        "__setup_namespace__": {},
        "inputs": "x",
    }
    cache = Cache(state)
    cache["foo"] = 1
    # Cache is empty when iterated despite all the framework keys
    assert list(cache) == ["foo"]
    # Internal keys still in state, untouched
    assert state["__event_log__"] == []
    assert state["inputs"] == "x"


# -----------------------------------------------------------------------------
# task_fail does not silence cache
# -----------------------------------------------------------------------------


def test_cache_writes_persist_through_task_fail():
    """A task_fail terminator still commits prior cache writes."""
    clear_agent_registry()
    config = connect_state(type="versioned", storage="memory")
    agent = Agent(name="t", llm=Dummy(), state=config)

    @agent.task
    def stash_then_fail() -> str:
        """Cache something, then fail."""
        pass

    @agent.task
    def recall() -> int:
        """Recall."""
        pass

    agent.llm.responses = [
        make_response(
            thinking="cache and fail",
            code='cache["x"] = 7\ntask_fail("intentional")',
        )
    ]
    with pytest.raises(TaskFail):
        stash_then_fail(session="s")

    agent.llm.responses = [
        make_response(thinking="recall", code='task_success(cache["x"])')
    ]
    assert recall(session="s") == 7
