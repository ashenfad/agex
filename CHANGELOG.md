# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.9.1] - 2026-03-12

### Added
- **PyfetchOpenAI Client**: Browser-compatible LLM client using `pyodide.http.pyfetch` for direct browser-to-API calls without a server proxy. Supports OpenRouter, OpenAI, and any OpenAI-compatible endpoint. Async-only with SSE streaming.
- **SSE Parser**: Standalone Server-Sent Events line parser (`agex.llm.sse`) for streaming LLM responses over pyfetch
- **Skills system**: `agent.skill()` API for registering skill documentation (YAML frontmatter with name, description, modules). Skills are mounted read-only at `/skills/<name>/SKILL.md` and listed in the system prompt. Module annotations link registered modules to their related skill docs.
- **Prompt caching**: `cache_control` breakpoints on system message and second-to-last conversation message for OpenRouter (Anthropic, Gemini). Extended cache TTL to 1 hour for `pyfetch_openai`.
- **Task-level chaptering**: Replaces event-level chaptering — agents chapter entire tasks using task boundaries. `ChapterEvent` stores event refs instead of copies for reversibility. Auto-triggers based on `chaptering_trigger` token threshold. Async chapter task supports async-only LLM providers.
- **`parent_ref` on `BaseEvent`**: Events automatically track their parent task for grouping.
- **`estimate_log_tokens`**: Tiktoken-based context size estimation for chaptering decisions.
- **IndexedDB storage backend**: Persistent versioned state in browser contexts via kvgit's IndexedDB backend.

### Changed
- **kvgit 0.1.5 Alignment**: Renamed imports to match kvgit API (`VersionedKV`)
- **Live State**: Moved `Live` from kvgit into `agex.state` (no longer imported from kvgit)
- **Filesystem Config**: Inlined `connect_fs` and config dataclasses from monkeyfs
- **Dependencies**: bumped kvgit>=0.1.7, monkeyfs>=0.1.2

### Removed
- **`GCVersionedKV`**: Replaced by kvgit's `clean_orphans()` on `VersionedKV`. Removed `high_water_bytes` and `low_water_bytes` from `StateConfig` and `connect_state`.

### Fixed
- **Pyodide Compatibility**: Platform guard for emscripten (lazy `HTTP` import), `is_function_body_empty` returns `True` when `inspect.getsource` is unavailable, tiktoken made optional (CORS-blocked encoding files).
- **Sandbox**: Unconditionally use sandbox as context manager; `ProcessSandbox` context manager fix
- **Session Validation**: Validate session IDs and require `IsolatedFSConfig` root
- **UnpicklableMarker**: `safe_commit` skips re-staging keys that are `UnpicklableMarker`s
- **Error messages**: Show real exception type instead of generic "Evaluation error"; HTTP status checking in pyfetch; last error surfaced in `TaskTimeout`.
- **Async rendering**: `async def` shown in capability signatures for coroutine functions so agents know to use `await`.
- **Chaptering**: Only triggers between tasks (not mid-task); chapter events emitted through `on_event` for live UI updates; correct timestamp ordering; filters `__chapter__` meta-events from index; skips gracefully when nested inside an async event loop.
- **Skill frontmatter**: Handles multiline YAML descriptions (block scalars, indented continuation).

## [0.9.0] - 2026-02-27

### Added
- **Process Isolation**: New `isolation` parameter for agents — run sandboxed code in a forked subprocess with crash protection (`isolation="process"`) or kernel-level restrictions (`isolation="kernel"`)
- **Sandtrap Integration**: Migrated sandbox execution to [sandtrap](https://github.com/ashenfad/sandtrap) for policy enforcement, AST rewriting, and serializable wrappers
- **MonkeyFS Integration**: Filesystem interception via [monkeyfs](https://github.com/ashenfad/monkeyfs) — supports `IsolatedFS` and `VirtualFS`
- **Termish Integration**: Virtual terminal via [termish](https://github.com/ashenfad/termish) with archive commands (tar, gzip, zip, unzip) and grep context flags
- **KVGit Integration**: Versioned state via [kvgit](https://github.com/ashenfad/kvgit)
- **Reprobate Integration**: Budget-controlled repr via [reprobate](https://github.com/ashenfad/reprobate)
- **LLM Client Overhaul**: Idiomatic SDK usage for LLM clients
- **Hierarchical Isolation**: Sub-agent calls work across process boundaries; nested process isolation rejected at registration time
- **St\* Reactivation**: `StFunction`/`StClass`/`StInstance` wrappers automatically reactivated after crossing process or remote host boundaries

### Changed
- **BREAKING**: Sandbox engine replaced — `sblite` removed in favor of `sandtrap`
- **BREAKING**: Filesystem layer replaced — custom VFS removed in favor of `monkeyfs`
- **BREAKING**: Terminal replaced — `faketerm` removed in favor of `termish`
- **BREAKING**: State store replaced — `kvit` removed in favor of `kvgit`
- **Memory Limits**: Deferred to sandtrap (dropped `psutil` dependency)
- **Dependencies**: Switched to PyPI packages; removed `xxhash`; added lint CI

### Fixed
- **Modal Image Build**: Use `add_local_dir` (SDK 1.3 removed `copy_local_dir`)
- **Sub-Agent Calls**: Auto-await async sub-agent task calls from sandbox code; use sync task loop for sub-agent calls
- **MacOS Archive Extraction**: Skip AppleDouble files (`._*`)

## [0.8.6] - 2026-01-02

### Added
- **State Init Variables**: Variables from `connect_state(init={...})` are now available in the agent's execution namespace

### Changed
- **Per-Task Snapshots**: State is now only snapshotted on task completion (success, fail, clarify, cancelled)

## [0.8.5] - 2025-12-30

### Added
- **State Reversion**: New `state.revert_to(commit_hash)` method to rollback agent state to a previous point
- **Event Commits**: Terminal events (`SuccessEvent`, etc.) now capture the commit hash *after* the result is snapshotted
  - Enables "revert to this result" workflows using `state.revert_to(event.commit_hash)`

## [0.8.4] - 2025-12-29

### Fixed
- **Cancellation Persistence**: `CancelledEvent` and preceding events now correctly persisted to disk
  - Moved `snapshot()` after cancellation event is added to log in sync/async loops
- **Stale Cancel Signals**: Cancel signals from previous task runs no longer affect subsequent tasks
  - Clears any pre-existing cancellation sentinel at task start
- **Class Registration Rendering**: Type hints in registered classes now show clean names
  - Strips module prefixes from type annotations (e.g., `ResponsePart` instead of `agex_ui.core.responses.ResponsePart`)
  - Applies to class docstrings, `__init__` signatures, and attribute types

## [0.8.3] - 2025-12-29

### Added
- **Modal Local Packages**: Automatic support for local Python packages in Modal deployments
  - Uses Modal's `add_local_python_source()` for packages in editable mode or local directories
  - `Dependencies` dataclass now tracks `local_packages` separately from PyPI packages

## [0.8.2] - 2025-12-28

### Added
- **Task Cancellation**: Cancel running tasks from another thread or process
  - `task.cancel(session="default")` writes sentinel to shared state
  - Task loop checks for sentinel at iteration boundaries
  - Works with `Live` (in-process) or `Versioned` disk (cross-process)
  - `TaskCancelled` exception with `iterations_completed` count
  - `CancelledEvent` recorded in event log
- **Documentation**: Task cancellation section in `task.md`, `agent.state()` in `agent.md`

### Changed
- **Modal Host**: Full support for `agent.state()` client-side access

## [0.8.1] - 2025-12-27

### Added
- **Modal Host**: New `agex.host.modal` module for severless agents
- **Modal Storage**: Two-tier state persistence for Modal host
  - `memory` storage using Modal Dict (7-day TTL, auto-named from fingerprint)
  - `disk` storage using Modal Dict + Volume (permanent, requires path)
- **`agent.state()`**: Inspect runtime state for debugging (local execution only)

## [0.8.0] - 2025-12-25

### Added
- **Remote Execution**: New `agex.host` module for distributed compute
  - `connect_host()` helper with `Local` and `HTTP` host implementations
  - `@remote` decorator for deploying agents to remote servers
  - Built-in FastAPI server (`agex.server`) for hosting agents
  - Support for serializing agents, state, and tasks across process boundaries
- **Host Configuration**: Agent-level `host` parameter to control where tasks execute
- **LLM Configuration**: New `connect_llm()` helper for streamlined LLM client setup
- **State Configuration**: New `connect_state()` helper for state initialization
- **Documentation**: Added comprehensive API docs for [Host](docs/api/host.md) and [LLM](docs/api/llm.md)

### Changed
- **BREAKING**: State configuration moved from task-level to agent-level
  - Before: `my_task(state=my_state)`
  - After: `agent = Agent(state=connect_state(...))` then `my_task(session="user1")`
  - Tasks now accept `session` parameter instead of `state` for session-scoped isolation
- **Agent Interface**: Simplified agent configuration with three main parameters: `llm`, `host`, and `state`
- **State Resolution**: State is now resolved by the host based on agent configuration and session ID
- **Task Signature**: Removed `state` parameter from task functions (use `session` instead)

### Removed
- **Removed**: `task.stream()` method for real-time event streaming

## [0.7.1] - 2025-12-22

### Fixed
- **Async Sub-Agent Timeouts**: Fixed timing accounting for async sub-agent calls in `TaskProxy`, ensuring parent agents correctly pause their timeout timer while awaiting child tasks

## [0.7.0] - 2025-12-22

### Added
- **Async Support**: Full async support for tasks and examples, with `async def` task definitions
- **Gemini Enhancements**:
  - `url_context` support for passing URLs to the model
  - `google_search` tool integration (renamed from `search_grounding`)
  - Prefill support for better steering
  - Specific fix for streaming responses
- **Advanced Prompting**:
  - `peek` and `forefront` strategies for user functions
- **Google GenAI**: Migration to the `google-genai` SDK

### Changed
- **Timeouts**: Explicit timeouts for Eval and LLM calls
- **Documentation**: Updated tone to be more moderate

### Fixed
- **Definition Lookup**: Fixed definition lookup on `cached_property`
- **Summarization**: Ensure system message is included when summarizing

## [0.6.0] - 2025-12-15

### Added
- **Versioned State**:
  - `KVStore.cas` for Compare-And-Swap operations
  - Versioned branching and conflict resolution strategies
- **Garbage Collection**: Orphaned task garbage collection (`GCVersioned`)

### Changed
- **Documentation**: Clarified task functions and directory structure help

## [0.5.1] - 2025-12-07

### Fixed
- **Namespaces**: Fixed summarization and garbage collection for namespaced state

## [0.5.0] - 2025-12-07

### Added
- **Summarization**: Event log summarization and specific coaching instructions
- **Tiered Context**: Support for tiered context integration

### Fixed
- **Events**:
  - Emit summary events to `on_event`
  - Fixed event token count defaults and detail thresholds
  - Garbage collection restricted to orphaned events only

[0.9.0]: https://github.com/ashenfad/agex/releases/tag/v0.9.0
[0.8.6]: https://github.com/ashenfad/agex/releases/tag/v0.8.6
[0.8.5]: https://github.com/ashenfad/agex/releases/tag/v0.8.5
[0.8.4]: https://github.com/ashenfad/agex/releases/tag/v0.8.4
[0.8.3]: https://github.com/ashenfad/agex/releases/tag/v0.8.3
[0.8.2]: https://github.com/ashenfad/agex/releases/tag/v0.8.2
[0.8.1]: https://github.com/ashenfad/agex/releases/tag/v0.8.1
[0.8.0]: https://github.com/ashenfad/agex/releases/tag/v0.8.0
[0.7.1]: https://github.com/ashenfad/agex/releases/tag/v0.7.1
[0.7.0]: https://github.com/ashenfad/agex/releases/tag/v0.7.0
[0.6.0]: https://github.com/ashenfad/agex/releases/tag/v0.6.0
[0.5.1]: https://github.com/ashenfad/agex/releases/tag/v0.5.1
[0.5.0]: https://github.com/ashenfad/agex/releases/tag/v0.5.0
