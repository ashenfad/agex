# Contributing to agex

## Setup

agex uses [uv](https://github.com/astral-sh/uv). From a clone:

```bash
uv sync
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

`uv sync` installs the package and the `dev` group (pytest, ruff,
pyright, pre-commit). Add provider SDKs with extras when you need them,
e.g. `uv sync --extra anthropic`, or `--extra all`.

## Checks

```bash
uv run --all-extras pytest   # tests, provider SDKs included
uv run ruff check .          # lint (pre-commit runs it, with ruff format)
uv run pyright               # types (pre-commit runs it before a push)
```

CI runs all three on every pull request: tests on Python 3.10-3.14,
lint and types once. Without the extras, the tests that need a
provider's SDK skip.

Live tests call a real model and are deselected unless asked for. They
need `OPENROUTER_API_KEY`, and `AGEX_LIVE_MODEL` picks the model
(`openrouter:meta/muse-spark-1.3-contributor` by default):

```bash
uv run --extra openrouter pytest -m live
```

## Commits and changes

- Conventional Commits (`feat:`, `fix:`, `docs:`, ...).
- A user-visible change gets a CHANGELOG line under Unreleased: a bold
  lead and one short sentence.
