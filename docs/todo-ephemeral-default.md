# TODO: Make Ephemeral State the Default

## The Problem: "Fake Persistence" with Real Constraints

Currently, `agex` has a design inconsistency. When users call a task without explicitly passing state:

```python
# No state parameter passed
result = my_task(data)
```

The system internally creates `Versioned(Memory())` and enforces:
- ✅ Full pickle safety checks on all objects
- ✅ Snapshotting after every iteration
- ✅ All persistence overhead and constraints
- ❌ But the state is discarded immediately after the task

This creates **all the constraints of persistence with none of the benefits**. Users hit confusing pickle errors for state that isn't even being persisted.

## The Current Behavior is Illogical

| Call Pattern | Internal State | Persistence | Constraints | Problem |
|--------------|----------------|-------------|-------------|---------|
| `my_task(data)` | `Versioned(Memory())` | ❌ None | ✅ Full | Constraints without benefits |
| `my_task(data, state=v)` | User's `Versioned` | ✅ Real | ✅ Full | Logical ✅ |

The first row is problematic: we're making users pay the cost of persistence constraints while providing no actual persistence.

## Proposed Solution: Ephemeral Default

Change the internal default to `Ephemeral()` when no state is passed:

```python
# No state passed → Use Ephemeral() internally → No constraints
result = my_task(data)  # Unpickleable objects work fine

# State passed → Use actual state → Appropriate constraints  
state = Versioned(Disk(...))
result = my_task(data, state=state)  # Pickle safety enforced
```

## Benefits

### **Logical Consistency**
- **Ephemeral behavior** gets **ephemeral constraints** (none)
- **Persistent behavior** gets **persistent constraints** (pickle safety)

### **Better User Experience**
- New users can return complex objects without mysterious errors
- Simple use cases "just work" without understanding persistence
- Advanced users explicitly opt into constraints when they want benefits

### **Performance**
- No unnecessary snapshotting for ephemeral tasks
- No pickle validation overhead when state isn't persisted
- Faster execution for simple scenarios

## Proposed User Experience

```python
# This should work - no persistence, no constraints
@agent.task
def get_database_cursor() -> sqlite3.Cursor:  # type: ignore[return-value]
    """Get a database cursor."""
    pass

cursor = get_database_cursor(data)  # ✅ Works fine

# This should enforce constraints - user wants persistence
state = Versioned(Disk("/path"))
cursor = get_database_cursor(data, state=state)  # ❌ Pickle error (expected)
```

## Implementation Details

This change requires:

1. **Modifying the default state factory** in `agex/agent/loop.py` to create `Ephemeral()` instead of `Versioned(Memory())`

2. **Conditional constraint checking** in `agex/eval/statements.py` to only enforce pickle safety when using persistent state types

3. **Adding state type detection** via a `get_root_state()` method to identify the underlying state type even when wrapped by decorators like `Namespaced` or `Scoped`

## Previous Implementation Challenges

A previous attempt was reverted due to:

### **Agent Resolution Issues**
- `RuntimeError: No agent found with fingerprint ...` in tests
- Suggests agent/task registration depends on `Versioned` state assumptions
- Need to decouple agent resolution from state persistence

### **Evaluator Integration Problems**  
- `AttributeError: 'Agent' object has no attribute '__enter__'`
- Indicates evaluator has assumptions about state type when handling `Agent` objects
- Particularly problematic in ephemeral contexts where agents are passed as values

### **Architectural Dependencies**
- Multiple components assume `Versioned` state is always available
- Need careful refactoring to separate execution state from persistence state
- Type annotation issues (`from __future__ import annotations`) from circular dependencies

## Success Criteria

The implementation should achieve:

- ✅ `my_task(data)` works with any Python object (no pickle constraints)
- ✅ `my_task(data, state=versioned)` enforces pickle safety  
- ✅ All existing tests pass
- ✅ Agent resolution works in both ephemeral and persistent modes
- ✅ No performance regression for persistent use cases

## Why This Matters

This change removes the biggest source of new-user friction (unexpected pickle errors) while preserving the power of persistence for users who want it. It makes `agex` behavior match user expectations: **pay persistence costs only when getting persistence benefits**. 