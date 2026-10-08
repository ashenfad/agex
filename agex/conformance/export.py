"""Write the shape corpus as JSON, with its JSON Schema, from the Python
sources.

    python -m agex.conformance.export

One JSON file per scenario under ``json/``, and under ``schema/`` the
JSON Schema of the scenario format. agex-ts reads these.
``tests/test_shape_corpus.py`` fails when the committed files differ
from what this would write.
"""

from __future__ import annotations

import json
from pathlib import Path

from nontainer.conformance.codec import dumps, json_schema

from .scenarios import SCENARIOS
from .shape import TaskScenario

__all__ = ["JSON_DIR", "SCHEMA_DIR", "files", "write"]

HERE = Path(__file__).resolve().parent
JSON_DIR = HERE / "json"
SCHEMA_DIR = HERE / "schema"


def files() -> dict[Path, str]:
    """Every file the export writes, by path, with its content."""
    out = {JSON_DIR / f"{s.name}.json": dumps(s) for s in SCENARIOS}
    schema = json_schema(TaskScenario, title="TaskScenario")
    out[SCHEMA_DIR / "task_scenario.schema.json"] = (
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
