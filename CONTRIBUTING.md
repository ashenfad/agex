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
uv run pytest            # tests
uv run ruff check .      # lint (pre-commit runs it, with ruff format)
uv run pyright           # types (pre-commit runs it before a push)
```

CI runs all three on every pull request: tests on Python 3.10-3.14,
lint and types once.

## Commits and changes

- Conventional Commits (`feat:`, `fix:`, `docs:`, ...).
- A user-visible change gets a CHANGELOG line under Unreleased: a bold
  lead and one short sentence.
