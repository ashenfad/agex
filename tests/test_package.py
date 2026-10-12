"""The package imports, and so do the pieces of its dependencies it is
built on."""

import importlib.metadata


def test_agex_imports():
    import agex

    assert set(agex.__all__) == {
        "Agent",
        "Outcome",
        "Session",
        "agent_tasks",
        "cancel_tasks",
        "stand_in_tasks",
        "NeedsInput",
        "Task",
        "TaskError",
        "TaskFailed",
        "TaskInterrupted",
        "TaskSpec",
    }
    assert importlib.metadata.version("agex").startswith("0.13")


def test_the_stubs_import_without_the_loop():
    """A task's worker imports ``agex.stubs``, and nothing more."""
    import subprocess
    import sys

    probe = (
        "import sys, agex.stubs\n"
        "print(sorted(m for m in ('agex.agent', 'agex.task', 'pydantic_ai') "
        "if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"
    import agex

    assert agex.Agent.__module__ == "agex.agent"
    assert agex.TaskSpec.__module__ == "agex.task"


def test_each_name_is_itself_whatever_was_imported_first():
    """No public name shares its module's name, so a module loaded first
    (by ``stand_in_tasks``, say) can't take a name's place on the
    package."""
    import subprocess
    import sys

    import agex

    for name in agex.__all__:
        probe = (
            "import importlib, agex\n"
            "for home in sorted(set(agex._HOMES.values())):\n"
            "    importlib.import_module(home, 'agex')\n"
            f"print(type(getattr(agex, {name!r})).__name__)"
        )
        out = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() != "module", name


def test_the_dependencies_agex_builds_on_are_there():
    from nontainer import Store
    from nontainer.conformance.harness import SCENARIOS
    from nontainer.turns import Turn, TurnEvent  # noqa: F401
    from pydantic_ai import direct  # noqa: F401

    assert SCENARIOS
    with Store(memory=True) as store:
        ws = store.open("probe")
        with ws.turn("r1"):
            pass
        ws.close()
