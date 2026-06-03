# Spawn type-sharing: sandbox-defined classes flow into clones

!!! note "Status: design — grounded against source, depends on `spawn`"

    Captured from discussion and verified against agex/sandtrap source (the
    `StClass` round-trip, the auto-activation hook, the `RemoteCache`
    cross-process reactivation model, and the dataclass structural-validation
    path). Builds on [`spawn.md`](spawn.md) (shipped). Targets the **agent's
    own** sandbox-defined types crossing into spawn clones — *not* host-level
    implicit registration of types in a `@agent.task` signature, which is a
    separate (security-sensitive) follow-up (see Non-goals).

## Summary

Let an agent define a class in its sandbox and use it in a `@spawn.task`
signature with **zero registration thought**:

```python
@dataclass
class Tile:
    name: str
    svg: str

@spawn.task
def make_tile(prompt: str) -> Tile:
    """Produce a 64x64 tile for the prompt."""
    pass

tile = make_tile("a small castle")   # a real Tile, attributes intact
```

Today this fails: a spawn clone is a **separate sandbox**, so `Tile` — defined
in the parent's sandbox — isn't in the clone's namespace. The clone can't
construct or return one (`NameError: name 'Tile' is not defined` → the clone
spins to a `TaskTimeout`). This design makes the agent's sandbox-defined types
referenced in a spawn signature travel into the clone automatically, reusing
the **exact machinery agex already uses to move sandbox-defined types across
the process-isolation boundary**.

## Why

- **Spawn's whole pitch is ergonomics**, and "but you can't return your own
  type" is a sharp corner. The agent shouldn't have to know that a clone is a
  different sandbox, or think about registration.
- **The boundary is not new.** A spawn clone (in-process, fresh namespace) is
  the same cross-sandbox boundary process isolation already crosses. The
  enabling machinery (`StClass` AST round-trip + reactivation) exists and is
  tested (`tests/agex/test_isolation_e2e.py::test_cached_stfunction_round_trips_across_processes`).
  We are wiring an existing capability into the spawn path, not inventing one.

## 1. The mechanism

A sandbox-defined class is an `StClass` (sandtrap) that stores its rewritten
AST so it can be serialized and recompiled. Its lifecycle (`sandtrap/wrappers.py`):

- **Active** = `_compiled_cls`/`_gates`/`_sandbox` are set (bound to the sandbox
  that defined it).
- `__getstate__` **clears** those fields → an **inactive** wrapper carrying only
  the AST + frozen refs.
- `activate(gates, *, sandbox, namespace)` recompiles the AST against a target
  sandbox's gates → binds it to **that** sandbox.

The sandbox auto-activates wrappers in its exec namespace (`Sandbox._auto_activate`),
but **only inactive ones** — an already-active wrapper is skipped. So:

> A parent's *active* `StClass` dropped into the clone's namespace is **skipped**
> by auto-activate and stays bound to the *parent's* gates — a policy/context
> leak. It must be round-tripped to **inactive** first, then the clone's
> auto-activate binds it to the clone's gates.

This is precisely what `RemoteCache.__getitem__` does on every cross-process
read (`agex/cache.py:237-251`): the value arrives inactive (pickled) and is
reactivated with the worker's local gates. **Spawn replicates that model**, in-process:

```
spawn.task(fn):   collect sandbox-defined StClasses referenced in fn's signature
spawn._run_one:   for each, produce a FRESH inactive copy (getstate/setstate)
                  inject the inactive copies into the clone's exec namespace
clone exec:       _auto_activate binds them to the clone's gates → the agent
                  can construct `Tile(...)` natively, under the clone's policy
```

## 2. Detecting the types

At `spawn.task(fn)` definition, walk the function's annotations — the return
annotation **and** each parameter annotation — and collect every value that is
an `StClass`. Recurse through generics (`get_origin`/`get_args`, as
`agex/eval/validation.py` already does) so `list[Tile]`, `dict[str, Tile]`,
`Tile | None` are covered. The annotation value is already the `StClass` object
(verified: `inspect.signature(fn).return_annotation` → `<StClass 'Tile'>`), so
no name resolution is needed.

Store the collected classes on the `SpawnTaskWrapper` (alongside the other
captured contract values), keyed by name.

## 3. Freshness — round-trip per invocation

The inactive copy must be **fresh per clone invocation**, not reused: once a
clone activates a wrapper it becomes active, and the next invocation's
auto-activate would skip it (re-binding to a stale run). Capture the inactive
**pickle bytes once** at definition; `pickle.loads` a fresh inactive copy per
`_run_one`. This mirrors process isolation (every cache read reactivates) and
keeps concurrent fan-out safe — each clone gets its own freshly-bound class.

Injection channel: seed the clone's per-invocation state
`__setup_namespace__` (the existing "names available in the first emission"
slot, surfaced by `build_namespace`) with `{name: inactive_copy}`. The clone's
auto-activate does the rest.

## 4. Validation — dataclasses work today; plain classes need leniency

Return-type validation (`validate_with_sampling`, `agex/eval/validation.py`)
behaves differently by shape:

- **Dataclasses → already cross-identity safe.** For a top-level dataclass
  annotation, validation accepts a value whose class has the **same `__name__`
  and field set**, even at a different identity (`validation.py:78-89`, comment:
  *"handles pickled dataclasses that have different identity but same
  structure"*). The clone's reconstructed `Tile` is a distinct object from the
  parent's, but structurally identical → **passes for free**. This is the v1
  target and the case the agent reaches for.
- **Plain (non-dataclass) classes and sandbox classes nested in generics →
  strict.** These fall to Pydantic `TypeAdapter` with `STRICT_CONFIG`
  (`validation.py:94-101, 114-115`), i.e. strict `isinstance`, which fails
  across reconstructed identities.

Two ways to cover the strict cases (deferred past v1 — see Open questions):
(a) extend the structural match to sandbox-defined non-dataclasses (compare by
name + public attribute/method set); or (b) reconstruct the return annotation
to reference the clone's reactivated classes so `isinstance` matches. (a) is the
smaller, more general change.

## 5. Params vs. return — an asymmetry

- **Return type** is the hard case: the clone must **construct** an instance, so
  it needs the class in its namespace (§1).
- **Parameter types** are easier: the instance is built **parent-side** (
  `_run_one` calls `task._bind_and_validate(*args)` in the parent context) and
  passed into the clone's loop as `inputs_instance`. The clone only **reads** it
  (`inputs.tile.name`), and cross-sandbox attribute reads already work (sandtrap
  permits public attribute access on any object). So a param's class need only
  be injected if the clone references it *by name* (e.g. an `isinstance` check).
  v1 injects both (return + params) for symmetry and to support such references;
  the load-bearing one is the return type.

## 6. The type-identity caveat (state it plainly)

The clone's reconstructed `Tile` is a **distinct class object** from the
parent's. Consequences:

- Reading attributes of the returned instance in the parent (`tile.name`) —
  **works** (duck-typed, cross-sandbox attribute reads are allowed).
- `isinstance(tile, Tile)` in the **parent** against the parent's `Tile` —
  **False** (different identity). This is an edge case (most callers read or
  forward the result); document it. Dataclass *validation* sidesteps it via
  structural matching, so the common path is unaffected.

## Decisions

- **Scope to the agent's own sandbox-defined types** referenced in a spawn
  signature. Host-level implicit registration is a separate follow-up (Non-goals).
- **Round-trip to inactive, then let the clone's auto-activate bind it** — never
  inject an active parent wrapper (gate leak). Mirror `RemoteCache`.
- **Fresh inactive copy per invocation** (cache the pickle bytes once).
- **v1 supports dataclass return types** (structural validation already cross-
  identity safe). Plain-class / generic-nested return types are an extension
  (validation leniency).
- **Inject return + param types** into the clone; the return type is required,
  params are for by-name references.

## Open questions

- **Validation leniency for non-dataclass sandbox classes** — extend the
  structural match (name + attr/method set), vs. annotation reconstruction.
  Pick (a) structural; confirm it composes with the generic/Pydantic path.
- **Nested/referential classes** — a `Tile` whose field is another sandbox
  class `Color`. The `StClass` AST round-trip carries frozen refs and
  auto-activates dependencies (`_activate_inner` walks `_frozen_refs`); confirm
  this transitively reconstructs referenced sandbox classes, or collect the
  transitive closure explicitly.
- **Cost** — pickling each signature class per invocation. Likely negligible
  (small ASTs); measure under fan-out and cache if needed.

## Non-goals

- **Not host-level implicit registration.** Auto-registering host classes that
  appear in a `@agent.task` signature is a separate decision with a security
  dimension (silently widening the agent's surface). This design only flows the
  agent's *own* sandbox-defined types into clones, which is always safe (the
  agent already holds them; clone ⊆ parent).
- **Not strict cross-sandbox type identity.** The clone's class is a distinct
  object; we make structural use work, not `is`-identity.
- **Not a change to sandtrap.** Reuses the existing `StClass` round-trip and
  auto-activation; the wiring lives in agex's spawn path.

## Implementation sketch (change-list)

- `agex/agent/spawn.py`
  - `SpawnTaskWrapper`: at construction, walk the wrapped task's signature
    annotations (return + params, recursing generics) for `StClass` instances;
    store `name → pickle.dumps(inactive copy)`.
  - `Spawn._run_one`: `pickle.loads` a fresh inactive copy of each; merge into
    the clone state's `__setup_namespace__` before `_run_task_loop`.
- `agex/eval/validation.py` (extension, post-v1): broaden the structural
  dataclass match to sandbox-defined non-dataclasses, or add a sandbox-class-
  aware lenient branch.
- Primer: drop the interim "use shared types in spawn signatures" caveat once
  this lands; until then, keep it.

## Verification

- **Dataclass return (v1 core):** agent defines `@dataclass class Tile`, returns
  it from a `@spawn.task`; assert the parent receives an object with the right
  fields/values. Mirror `tests/agex/agent/test_spawn.py`.
- **Plain class return:** agent defines a plain class, returns it — assert it
  works (after the validation-leniency extension) or documents the limitation.
- **Param of a sandbox type:** pass a parent-constructed `Tile` into a
  `@spawn.task`; the clone reads `inputs.tile.name`.
- **Generics:** `-> list[Tile]` round-trips.
- **Concurrency:** `spawn.map` over a `Tile`-returning task — each clone gets a
  freshly-bound class (no active-wrapper reuse across invocations).
- **No gate leak:** assert the clone runs under the *clone's* policy, not the
  parent's, when constructing the injected class (e.g. a scoped/registered
  capability difference is observed correctly).
