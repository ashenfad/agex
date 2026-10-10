"""Write the shape corpus as JSON, with its JSON Schema, from the Python
sources.

    python -m agex.conformance.export

One JSON file per scenario under ``json/`` (a task scenario, or a code
scenario, named ``code-...``), and under ``schema/`` the JSON Schema of
each scenario format. agex-ts reads these.
``tests/test_shape_corpus.py`` fails when the committed files differ
from what this would write.
"""

from __future__ import annotations

import json
from pathlib import Path

from nontainer.conformance.codec import dumps, json_schema

from .code_scenarios import CODE_SCENARIOS
from .scenarios import SCENARIOS
from .shape import CodeScenario, TaskScenario

__all__ = ["JSON_DIR", "SCHEMA_DIR", "files", "write"]

HERE = Path(__file__).resolve().parent
JSON_DIR = HERE / "json"
SCHEMA_DIR = HERE / "schema"


def files() -> dict[Path, str]:
    """Every file the export writes, by path, with its content."""
    out = {JSON_DIR / f"{s.name}.json": dumps(s) for s in SCENARIOS}
    out.update({JSON_DIR / f"{s.name}.json": dumps(s) for s in CODE_SCENARIOS})
    for kind, name in (
        (TaskScenario, "task_scenario"),
        (CodeScenario, "code_scenario"),
    ):
        schema = json_schema(kind, title=kind.__name__)
        out[SCHEMA_DIR / f"{name}.schema.json"] = (
            json.dumps(schema, indent=2, sort_keys=True) + "\n"
        )
    return out


def write() -> list[Path]:
    """Write :func:`files`, and remove scenario JSON no scenario makes
    any more; the paths written."""
    wanted = files()
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    for stale in JSON_DIR.glob("*.json"):
        if stale not in wanted:
            stale.unlink()
    for path, text in wanted.items():
        path.write_text(text)
    return sorted(wanted)


if __name__ == "__main__":
    for path in write():
        print(path.relative_to(HERE))
