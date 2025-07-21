# Deferred Improvements

This document tracks architectural improvements and tech debt that would be nice to address but are not blocking current development goals.

## PrintAction should store raw exceptions instead of pre-rendered strings

**Current behavior:**
```python
PrintAction([f"💥 Evaluation error: {e}"])
```

**Preferred behavior:**
```python
PrintAction(["💥 Evaluation error:", e])
```

**Why this would be better:**
- Consistent with late rendering philosophy (store raw objects, render just before LLM)
- Preserves full exception information including stack traces for debugging
- Allows renderer to decide how much detail to show LLM vs. developers
- More transparent for developers inspecting event logs

**Why deferred:**
- Would require updating ~10+ test files that expect pre-rendered strings
- Would need renderer changes to handle raw exceptions
- Current approach works fine and is not blocking streaming functionality
- Estimated effort: 1+ hours for marginal architectural consistency gain

**Priority:** Low - architectural nicety, not functional requirement 