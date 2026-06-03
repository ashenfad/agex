# Wrapped vs. raw: is sandtrap's wrapping still earning its keep?

!!! question "Status: open design question — not decided"

    A deliberate evaluation, not a plan. agex runs sandtrap in `mode="wrapped"`;
    this doc asks whether that's still the right default, captures what wrapping
    buys vs. costs, and frames the options. Triggered by the spawn work, where
    the wrapping leaked at every boundary (see [`spawn.md`](spawn.md),
    [`spawn-type-sharing.md`](spawn-type-sharing.md)). **Do not decide this
    under spawn's pressure** — it touches caching, isolation, and persistence.

## The question

sandtrap has two modes (`sandtrap/sandbox.py`): `"wrapped"` (default) wraps
sandbox-defined functions/classes/instances as `StFunction`/`StClass`/
`StInstance` so they can be **pickled**; `"raw"` returns plain objects. agex
sets `mode="wrapped"` in exactly two places — the main eval bridge
(`agex/eval/bridge/__init__.py:133`) and the terminal `python` command
(`agex/python_cli.py:213`). Is wrapped still the right call?

## What wrapping buys (the entire return)

Picklability of **agent-authored** functions/classes/instances, which enables:

1. **Cache-of-code** — `cache["fn"] = a_sandbox_helper`, reused across turns and
   tasks. Backed by the `__cache_wrappers__` index + `__sandtrap_activate__`
   reactivation in `agex/cache.py`. Tests: `test_cache.py::test_sandbox_function_storeable_within_task`,
   `::test_sandbox_function_round_trips_across_tasks`.
2. **Cross-process / kernel isolation** — sandbox-defined code/objects crossing
   the IPC boundary, reactivated in the worker. `agex/agent/task.py:142`
   (`_reactivate_result`); `test_isolation_e2e.py::test_cached_stfunction_round_trips_across_processes`.
3. **Versioned (kvgit) persistence** of agent-authored code/objects.

Plus one incidental: `agex/agent/utils.py` uses `isinstance(fn, StFunction)` to
detect "sandbox-defined, can't read source." The whole agex dependency surface
is four files: `cache.py`, `agent/task.py`, `agent/utils.py`, `agent/spawn.py`.

## What it costs

The wrapping leaks at **every boundary**, in **every mode** (including the
default in-process path, which gets none of the benefits above — in-process,
objects pass by reference; no pickling is needed):

- A host callable returned into the sandbox is re-wrapped as `StFunction`, so
  `spawn.submit(gen, …)` received a wrapper, not the task → the
  `Spawn._resolve_task` unwrap (`spawn.md`).
- `StClass` is **not a Python type**, so neither `isinstance` nor Pydantic can
  validate it (Pydantic only warns and passes everything) → the bespoke
  `StInstance`-identity validation branch (`spawn-type-sharing.md`).
- Cross-sandbox use needs round-trip-to-inactive + reactivation, and an
  *active* wrapper dropped into another sandbox silently leaks the original
  sandbox's gates.

## The reframe

- **The original driver is gone.** Wrapping was load-bearing when agex carried
  the whole REPL namespace across turns. Namespaces are turn-local now, so that
  reason no longer applies.
- **The value is concentrated in the margins.** Returning *data* (DataFrames,
  dicts, dataclasses-of-primitives) across processes never needed wrapping —
  data pickles fine. Wrapping is only for sandbox-defined *code* and *instances
  of sandbox classes*. And cache-of-code overlaps heavily with `helpers/`
  (source files in the VFS, recompiled on import — no wrapping needed for
  persistence). So the unique remaining value is "store/cross a live
  sandbox-defined callable or class instance," which is comparatively niche.
- **The cost is paid up front, everywhere.** Every execution pays the wrapping
  tax to enable a capability most executions don't use.

So there is a real case that wrapped mode is over-applied. spawn is evidence the
tax is concrete, not theoretical.

## Evidence: spawn type-sharing, wrapped vs. raw (measured)

Forcing `mode="raw"` and returning a sandbox-defined class from a task:

| | wrapped (today) | raw |
|---|---|---|
| returned object | `StInstance` wrapper | a real instance of the real class |
| `-> Tile` validation | bespoke `StClass`-identity branch | native `isinstance` |
| `-> list[Tile]` | **fails** (nested class unseeded → timeout; now a fail-fast) | native Pydantic validation, **free** |
| getting the class into a clone | pickle→inactive→seed→auto-activate | put the real class in the namespace |
| `isinstance(result, Tile)` in parent | False (reconstructed ≠ original) | True (same class object) |

So the entire spawn type-sharing apparatus (`_collect_seed_classes`, the
per-invocation round-trip, the validator branch, the generic fail-fast) exists
*only* to work around wrapping, and would largely **delete** under raw.

**Critical caveat that rules out wholesale removal:** process/kernel isolation
returns results across a process boundary **by pickle** — you can't pass real
objects by reference across processes. A raw sandbox-defined instance is
unpicklable, so *removing wrapping entirely would break returning sandbox-defined
types from process/kernel-isolated tasks* — a security headline, not a niche.
Wrapping (or some serialization) is genuinely required there.

This reframes the target: not "remove wrapping" but **"stop wrapping in-process;
serialize only at the boundary that needs it."**

## Options

- **A — keep wrapped (status quo).** Zero migration risk; keeps cache-of-code,
  cross-process code, versioned-code persistence. Keeps paying the boundary tax
  (mitigated case-by-case, as spawn did).
- **B — default to raw.** In-process becomes trivial (real functions/classes,
  real `isinstance`, no reactivation, no unwrap; spawn type-sharing would be a
  near-freebie — pass the real class to the clone). **Breaks** cache-of-code,
  cross-process code, and versioned-code persistence (sandbox-defined code/
  instances become unpicklable). Real tests fail; needs a "what relies on this"
  pass and a migration story. Note: raw mode is a first-class, tested sandtrap
  mode (`sandtrap/tests/test_sandbox_factory.py`, `test_wrappers.py`).
- **C — less-leaky wrapped (sandtrap-side).** Keep picklability, but stop the
  wrapping from leaking: make `StClass` behave type-like (support `isinstance`
  via a metaclass `__instancecheck__`, so the validator needs no special
  branch); don't re-wrap host objects returned into the sandbox (kills the
  `_resolve_task` unwrap); expose a public `unwrap`/identity helper.
- **D — per-context (raw in-process, serialize at the boundary).** Default to
  `raw` for in-process + ephemeral execution (the common path, and **every spawn
  clone** — clones are always in-process), and serialize sandbox-defined
  code/instances only where it's actually required: crossing a process/kernel
  boundary, or persisting to versioned state. This makes spawn type-sharing free
  and complete *and* keeps isolation/persistence working. The cost is a decision
  rule (which mode applies) and handling "a raw agent tries to cache/persist
  code" (clear error vs. on-demand serialize). **Current lean.**

## What to investigate before deciding

- **Real usage of cache-of-code and cross-process code** — how much of it is
  load-bearing vs. theoretically-supported? If it's rare, B gets cheaper.
- **Could raw + `helpers/` cover the reuse use case?** Agents reuse logic via
  `helpers/` (source) far more naturally than via cached function objects.
- **Per-context mode** — could clones (ephemeral, in-process, never persisted)
  run `raw` while the durable parent stays `wrapped`? (Doesn't fully solve
  cross-sandbox type-sharing, since the parent's class is still an `StClass`,
  but worth weighing.)
- **The cost of C** — how much sandtrap work to make wrapping non-leaky, and
  does a type-like `StClass` interact badly with anything?

## Recommendation

Don't fold this into spawn. Ship spawn and its (interim, brittle) type-sharing
within wrapped mode — done in PR #68 — then do this as an **immediate follow-on
PR**. Current lean: **D** (per-context: raw in-process, serialize only at the
process/persistence boundary). It keeps the one capability wrapping genuinely
earns — serialization where it's actually required (process/kernel isolation,
versioned persistence) — while making the common in-process path raw, which
deletes the spawn type-sharing apparatus and makes generics/params/identity work
natively. D subsumes the good parts of B (raw simplicity) without B's fatal flaw
(breaking process/kernel isolation), and may pull in pieces of C (a type-like
`StClass`) for the still-wrapped boundary. The follow-on should open with the
"what actually relies on picklable sandbox-defined code?" audit, then specify
the decision rule and the cache/`task.py`/`_reactivate_result` changes.
