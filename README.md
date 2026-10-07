# agex

**agex is being rebuilt.** 0.13 is a new API: a loop over
[nontainer](https://github.com/ashenfad/nontainer), where a session
drives a versioned workspace turn by turn and a task runs on a fork of
one and hands back a typed value. Nothing is usable yet. The design is
in [docs/design.md](docs/design.md) and the order of work in
[docs/plan.md](docs/plan.md).

The 0.12 line (library-friendly agents that work directly with your
existing Python codebase) is at the
[`v0.12.4`](https://github.com/ashenfad/agex/tree/v0.12.4) tag, and on
PyPI as `agex==0.12.4`.

## Development

```bash
uv sync
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md).
