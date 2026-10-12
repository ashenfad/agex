"""The examples run, on a scripted model, so none of them rots; and the
README's code blocks compile."""

import ast
import importlib.util
import re
import sys
import textwrap
from pathlib import Path

import pytest
from nontainer.conformance.corpus import calls, says

from agex import Agent
from agex.providers.scripted import ScriptedProvider

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"


def example(name):
    """The example module ``name``, imported from its file."""
    path = EXAMPLES / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"agex_example_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def runs(code):
    return calls("run_python", code=textwrap.dedent(code))


def test_the_session_example(capsys):
    agent = Agent(
        ScriptedProvider(
            [
                runs("open('fib.py', 'w').write('print([0, 1, 1, 2])')"),
                says("fib.py prints the first ten."),
                runs("open('fib.py', 'w').write('print([0, 1, 1, 2, 3])')"),
                says("Now twenty."),
            ]
        )
    )
    example("session").main(agent)
    out = capsys.readouterr().out
    assert "[completed] fib.py prints the first ten." in out
    assert "[completed] Now twenty." in out
    assert "print([0, 1, 1, 2, 3])" in out
    assert "2 runs" in out


def test_the_task_example(capsys):
    agent = Agent(
        ScriptedProvider(
            [runs("task.success(Recipe('Spinach omelette', ['beat', 'fry'], 15))")]
        )
    )
    example("task").main(agent)
    out = capsys.readouterr().out
    assert "Spinach omelette (15 min)" in out
    assert "  2. fry" in out


def test_the_task_example_says_when_the_task_fails(capsys):
    agent = Agent(ScriptedProvider([runs("task.fail('no oven')")]))
    example("task").main(agent)
    assert "no recipe:" in capsys.readouterr().out


def test_the_asks_example(capsys):
    agent = Agent(
        ScriptedProvider(
            [
                runs("task.needs_input('How many people are coming?')"),
                runs("task.success(Booking('Trattoria Quieta', 'Friday', 4))"),
            ]
        )
    )
    example("asks").main(agent)
    out = capsys.readouterr().out
    assert "asked: How many people are coming?" in out
    assert "booked Trattoria Quieta on Friday for 4" in out


LEAD = """\
    import json

    @agex.task
    def label(review: str) -> str:
        \"""positive, negative or mixed\"""

    reviews = json.load(open('reviews.json'))
    labels = label.map(reviews)
    counts = {k: labels.count(k) for k in ('positive', 'negative', 'mixed')}
    json.dump(counts, open('summary.json', 'w'))
"""


def test_the_agent_tasks_example(capsys):
    lead = [runs(LEAD), says("Wrote summary.json.")]

    def next_step(sent):
        # the lead's first message is the prompt; a helper's is its brief
        if len(sent) > 1 and sent[1].startswith("reviews.json"):
            return lead.pop(0)
        return runs("task.success('positive')")

    example("agent_tasks").main(Agent(ScriptedProvider(next_step)))
    out = capsys.readouterr().out
    assert "[completed] Wrote summary.json." in out
    assert '"positive": 4' in out


@pytest.mark.parametrize("doc", ["README.md", "docs/quick-start.md"])
def test_the_docs_code_compiles(doc):
    text = (ROOT / doc).read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert blocks
    for block in blocks:
        ast.parse(block)
