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

Live tests call real models and are deselected unless asked for. Each
provider's run when its key is set (`OPENROUTER_API_KEY`,
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`), on a small
model by default; `AGEX_LIVE_MODEL` (OpenRouter), `AGEX_LIVE_ANTHROPIC`,
`AGEX_LIVE_OPENAI` and `AGEX_LIVE_GOOGLE` pick others:

```bash
uv run --all-extras pytest -m live            # every provider with a key
uv run --all-extras pytest -m live -k google  # one of them
```

## Commits and changes

- Conventional Commits (`feat:`, `fix:`, `docs:`, ...).
- A user-visible change gets a CHANGELOG line under Unreleased: a bold
  lead and one short sentence.
