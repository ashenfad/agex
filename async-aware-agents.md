# Async-aware sub-agent calls

## Current state

Sub-agent task calls from sandbox code always use the **sync task loop**
(`_run_task_loop` + sync LLM client).  This avoids event-loop conflicts
when the orchestrator runs inside `aexec`, but means all sub-agent calls
are sequential — no `asyncio.gather`.

## Goal

Enable parallel sub-agent calls from sandbox code:

```python
# LLM-generated orchestrator code
import asyncio
data_a, data_b = await asyncio.gather(
    make_data("dataset A"),
    make_data("dataset B"),
)
```

## Why this works in principle

sandtrap's `aexec` wraps sandbox code in `async def __st_aexec__()`, so
`await` and `asyncio.gather` work natively.  The sub-agent's async task
wrapper returns a coroutine that runs on the main event loop — no new
loops, no thread conflicts.

## What needs to change

1. **`_wrap_sub_agent_task`** — detect whether there's a running event
   loop.  If yes (inside `aexec`), return the coroutine from the async
   task wrapper.  If no (inside `exec`), call the sync task func.

2. **Primer coaching** — agents that have async sub-agent fns need a
   primer addition telling the LLM to use `await` on sub-agent calls and
   that `asyncio.gather` is available for parallel dispatch.  Something
   like:

   > Sub-agent functions are async.  Always `await` them:
   > `result = await make_data("test")`.  To call multiple sub-agents in
   > parallel, use `asyncio.gather`:
   > `a, b = await asyncio.gather(make_data("x"), make_data("y"))`.

3. **`asyncio` availability** — `asyncio` must be importable in the
   sandbox.  Either register it as a policy module automatically when
   async sub-agent fns are present, or add `gather`/`sleep` as builtins.

## Considerations

- The `asyncio` module import needs to be allowed by sandtrap policy.
- `on_token` streaming from parallel sub-agents will interleave.
- Error handling: if one branch of a gather fails (TaskFail/TaskClarify),
  the other branches should be cancelled cleanly.
