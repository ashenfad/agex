"""The package imports, and so do the pieces of its dependencies it is
built on."""

import importlib.metadata


def test_agex_imports():
    import agex

    assert set(agex.__all__) == {"Agent", "Outcome", "Session"}
    assert importlib.metadata.version("agex").startswith("0.13")


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
