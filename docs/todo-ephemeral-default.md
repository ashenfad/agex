# TODO: Make Ephemeral State the Default

## Motivation

Currently, `agex` uses `Versioned` state by default. This is a powerful feature that provides conversational continuity by persisting state between turns and ensuring all data is serializable (via pickling).

However, for new users and simple use cases, this is often overkill. The requirement that all state be pickleable is a significant constraint and a source of friction. For example, a user cannot simply return a complex, non-pickleable object from a tool and have it "just work".

To improve the early-adopter experience and make `agex` more approachable, the default state should be `Ephemeral`. This state would be:

-   **In-memory:** No serialization overhead.
-   **By-reference:** Passes Python objects directly without modification or serialization.
-   **Unconstrained:** Works with any Python object, removing the pickleable requirement.

The powerful `Versioned` state would then become an opt-in feature for users who explicitly need persistence and conversational memory.

## Attempted Solution

We attempted to implement this by:

1.  **Creating `agex/state/ephemeral.py`:** A new, simple state class that stores values in a `dict`.
2.  **Changing the Default:** Modifying `agex.run()` (in `agex/agent/loop.py`) to use `Ephemeral` as the default `state_factory`.
3.  **Conditional Safety Checks:** Updating the evaluator (in `agex/eval/statements.py`) to only perform pickle-safety checks when the active state is an instance of `Versioned`.
4.  **Identifying Root State:** This required a way to check the type of the "root" state, even when wrapped by decorators like `Namespaced` or `Scoped`. We added a new abstract method, `get_root_state()`, to the `State` ABC and implemented it across all concrete and wrapper state classes.
5.  **Fixing Cascade of Failures:** This change triggered numerous test failures. The process involved:
    *   Adding `from __future__ import annotations` to several state files to resolve `NameError` exceptions from circular dependencies in type hints.
    *   Implementing `get_root_state()` in all `State` subclasses.
    *   Updating tests that specifically relied on the pickle-error behavior of the old `Versioned` default.
    *   Fixing a complex bug in the dual-decorator logic (`@agent.fn` on an `@agent.task`) where the state was not being correctly propagated from an orchestrator agent to the sub-agent task (`agex/agent/registration.py`).

## Blockers

The implementation was reverted because even after fixing the initial test failures, we were left with critical, unresolved errors:

*   **Agent Resolution Failure:** `RuntimeError: No agent found with fingerprint ...` in test environments. This suggests that switching the default state breaks the mechanism by which agents and tasks are registered and discovered during tests.
*   **Evaluator `__enter__` Failure:** `AttributeError: 'Agent' object has no attribute '__enter__'`. This indicates a fundamental issue in how the `agex` evaluator handles the `Agent` object itself when it's passed as a value, which is more common in an `Ephemeral` state world.

These errors point to deep architectural assumptions that `Versioned` state is always available. A future attempt will need to carefully disentangle the agent/task resolution logic from the state persistence mechanism. 