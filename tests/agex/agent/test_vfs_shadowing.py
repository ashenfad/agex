import pytest

from agex import Agent, clear_agent_registry, connect_fs
from agex.agent.emissions import FileWriteEmission
from agex.agent.events import SystemNoteEvent
from agex.llm import Dummy
from tests.agex._emissions import make_response


@pytest.fixture(autouse=True)
def cleanup():
    clear_agent_registry()
    yield
    clear_agent_registry()


def test_vfs_shadowing_warning():
    """Verify that creating a VFS file that shadows a system module emits a warning."""
    import json

    llm = Dummy(
        responses=[
            make_response(
                thinking="I will create a json.py file.",
                file_actions=[
                    FileWriteEmission(
                        path="json.py", content="def loads(s): return 'shadowed'"
                    )
                ],
                code="import json\ntask_success(json.loads('{}'))",
            )
        ]
    )

    agent = Agent(llm=llm, fs=connect_fs(type="virtual"))
    # Register real json module so it's in policy
    agent.module(json)

    events = []

    def on_event(ev):
        events.append(ev)

    @agent.task
    def task():
        """Run task."""
        pass

    # 1. Verify result (real json should be used)
    result = task(on_event=on_event)
    assert result == {}  # Real json.loads('{}') is {}

    # 2. Verify warning event was emitted
    warnings = [
        e for e in events if isinstance(e, SystemNoteEvent) and "shadows" in e.message
    ]
    assert len(warnings) > 0
    assert "json.py" in warnings[0].message
    assert "registered system module 'json'" in warnings[0].message


if __name__ == "__main__":
    pytest.main([__file__])
