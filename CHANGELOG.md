# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Added
- **`agent.terminal_command(handler, ...)`** — register a custom shell
  command for `terminal_action`.  Decorator-style; the handler
  receives a `TerminalContext` (args, stdin, stdout, fs) and returns
  `None` (success) or a `CommandResult`.  Mirrors `agent.fn`
  ergonomics: bare decorator, decorator-factory with options, and
  direct-call shapes all supported.
- **`agent.terminal_command_factory(name, factory, ...)`** — register
  a command via a factory closure for handlers that need agex
  per-action runtime context (state, vfs).  The factory is invoked
  once per `terminal_action` with a fresh `TerminalRuntime` and
  returns a termish `CommandFunc`.
- **`agex.terminal` module** — public types for handler authors:
  `TerminalContext`, `TerminalRuntime`, plus re-exports of termish's
  `CommandContext` / `CommandResult` / `CommandFunc` so handlers can
  import their full toolset from one place.

### Changed
- **`register_git` migrated to `terminal_command_factory`.** No
  behavior change — git still mounts the skill markdown and surfaces
  in `terminal_action` — but the wiring now goes through the public
  registration API instead of a hardcoded branch in
  `build_terminal_commands`.  This is the API sanity check: the
  existing complex case migrates cleanly.
- **`terminal_action` primer wording** updated to acknowledge that
  hosts may register additional commands and to recommend
  `<command> --help` for discovery.

### Reserved names
- The string `"python"` is reserved for agex's internal bridge to
  nested `python_action` execution.  `terminal_command(...)` and
  `terminal_command_factory(...)` raise `ValueError` if a registration
  attempts to use it.  All other names — including termish builtins
  (`ls`, `cat`, `grep`, ...) — follow termish's existing
  "user-injected commands override builtins" contract; user
  registrations are last-wins among themselves.


## [0.12.0] - 2026-04-29

### Changed (breaking)
- **Stateless namespace contract.** Each `python_action` now runs as a
  fresh Python script.  Variables, imports, and definitions defined in
  one action no longer carry into the next; the previous "persistent
  REPL" behaviour is gone.  Cross-action continuity goes through three
  explicit channels:
    - **`cache[k] = v`** — agent-session-scoped persistent dict
      injected into every emission's namespace (see below).
    - **Virtual Filesystem** — files written under `/scratch/` and
      reusable code under `helpers/*.py` imported in any later action.
    - **Event log** — the agent's record of what it has done, surfaced
      via chapter summaries and `<TERMINAL>` `cat` / `ls`.
  The primer is rewritten end-to-end to teach this contract.  Agents
  that previously relied on automatic variable persistence now have
  three deliberate, inspectable channels to choose from.

### Added
- **`cache`** — agent-session-scoped persistent dict, automatically
  available to the agent in every `python_action`.  Picklability is
  validated at write time using stdlib `pickle.HIGHEST_PROTOCOL` —
  the protocol the state codec uses underneath — so the agent gets
  an immediate `CacheError` rather than discovering a silent
  `UnpicklableMarker` later.  Sandbox-defined functions and classes
  (`StFunction` / `StClass`) define `__getstate__` / `__setstate__`
  and round-trip across tasks via sandtrap's `__sandtrap_activate__`
  container hook (sandtrap >= 0.2.0); lambdas and other
  locally-defined functions outside the sandbox don't pickle and are
  rejected at the validator.
- **Cache works the same in any isolation mode.** Under
  `isolation="process"` / `"kernel"`, `build_namespace` injects a
  `RpcProxyMarker` instead of a live `Cache`; the worker substitutes
  it with a `RemoteCache` (also in `agex.cache`) that forwards
  method calls to the parent over sandtrap's RPC channel
  (sandtrap >= 0.2.1).  Writes propagate to the parent's session
  cache; reads see whatever the parent has cached; cached
  sandbox-defined helpers round-trip and re-activate against the
  worker's gates on read.  The agent contract is identical across
  modes — the only difference is per-call IPC latency under
  isolation.
- `execute_sandboxed`, `aexecute_sandboxed`, and `run_file_in_sandbox`
  now return the post-exec namespace dict so one-shot callers can
  inspect what a script computed without round-tripping through state.

### Removed (breaking)
- `agex.state.safe_commit` → renamed to `agex.state.commit_state`.
  The `referenced_keys` parameter is gone (the mutation-detection
  re-stage path it drove no longer exists).
- `build_namespace` signature simplified to
  `(state, agent, agent_name, on_event=None) -> (namespace,
  injected_keys)`.  The `pre_keys` slot in the return tuple is gone
  (no deletion-detection layer to feed).
- `handle_result` no longer takes `pre_keys`, `pre_ids`, or
  `injected_keys`; signature is now
  `(result, state, agent_name, on_event=None, emission_id=None)`.
- The static-analysis mutation-detection layer
  (`collect_python_refs`, `accumulated_refs` plumbing, the
  `__sys_user_fn_names__` "User Defined Functions" forefront message)
  is removed — agex no longer needs `sandtrap.find_refs`.

### Internal
- State codec now uses `kvgit.codecs.scientific()` (the named preset)
  instead of constructing `compose(NumpyCodec())` inline.
  Functionally identical today; agex automatically picks up future
  scientific codecs (Arrow, etc.) when kvgit adds them.

### Dependencies
- `sandtrap >=0.2.1,<0.3.0` (was `>=0.1.15,<0.2.0`).  0.2.0 added
  the `__sandtrap_activate__` container hook used by `Cache`; 0.2.1
  added the worker→parent RPC channel used by `RemoteCache` for
  cross-process cache parity.
- `termish >=0.1.6,<0.2.0` (was `>=0.1.5,<0.2.0`).

### Migration
- Replace `from agex.state import safe_commit` with `commit_state`;
  drop any `referenced_keys=...` argument.
- Code patterns that relied on agent variables surviving across
  tasks need to be updated:
    - Python objects the agent wants to remember → `cache[...]`
    - Reusable code → `helpers/*.py` imported via `import helpers.foo`
- Tests that asserted state values written by sandbox code (e.g.
  `state.get("result")` after running an agent's code) should either
  capture the value via `task_success(value)` or read the namespace
  dict returned by `execute_sandboxed` / `run_file_in_sandbox`.

## [0.11.2] - 2026-04-28

### Added
- **Chunked codec for numpy/pandas state.** When numpy is installed,
  ndarrays and DataFrame block buffers are externalized as
  content-addressed chunks via `kvgit.codecs.NumpyCodec` — a 10 MB
  DataFrame with N derived slices stores ~10 MB instead of N×10 MB.
  Decode semantics match `pickle.loads`. Existing v2 sessions stay
  v2 until the first chunked write; no migration. Without numpy, no
  change.

### Changed
- **Prints batched into one `OutputEvent` per emission.** All `print()`
  calls from a `python_action` now collapse into a single
  `OutputEvent(parts=[...])` instead of one event per call. Consumers
  that count prints by `len(output_events)` should sum `PrintAction`
  parts across events.
- **`kvgit` constraint** bumped to `>=0.3.0,<0.4.0` for the chunked
  codec.

## [0.11.1] - 2026-04-28

### Added
- **Verify-before-commit primer guidance** — when a task returns a
  testable artifact, the primer nudges the agent to assert against
  known cases in the same `python_action` as `task_success`, so a
  broken result fails the turn instead of shipping.

### Changed
- **System-note rendering** — `SystemNoteEvent`s are wrapped with a
  `[system] ` prefix when rendered into the LLM's user-role message
  so framework telemetry stays distinguishable from user speech.
  Applied at the renderer (not the event boundary) so persisted logs
  stay clean.
- **Builtin primer rewrite** — fresh-eyes pass: fixes stale references
  (`edit_file` ops, "most stdlib available"), drops redundant sections,
  adds a first-class **Terminal** capability section, and clarifies
  that terminal/git act on the agent's own VFS. Net ~20% shorter
  (~8.7k → ~6.9k chars).

## [0.11.0] - 2026-04-24

**Retooling** — the agex turn shape now matches provider-native tool
use.  Motivated by Gemini 3 `thought_signature` requirements, Claude
4's interleaved extended thinking, and GPT-5-family routing through
the Responses endpoint.  Breaking: event log, primer, and wire format
are not compatible with 0.10.x.  See `agex/llm/formats/tool_use/README.md`
for the wire-format overview and `agex/llm/README.md` for per-
provider reasoning round-trip.

### Added
- **Emission-list `ActionEvent`** — `ActionEvent(emissions=[...])` is
  an ordered list of `PythonEmission` / `TerminalEmission` /
  `FileWriteEmission` / `FileEditEmission` / `TextEmission` /
  `ThinkingEmission`.  A single turn may carry several emissions;
  Python emissions share a namespace.
- **Native thinking by default** on every provider (`native_thinking=True`
  on `ToolUseWireFormat`).  Thinking arrives as provider-native
  content blocks (Claude extended thinking, Gemini thought parts,
  OpenAI Responses reasoning items, OpenRouter `reasoning_details`)
  and round-trips via `ThinkingEmission.signature` bytes with
  provider-specific tag prefixes.
- **OpenAI Responses API support** — `gpt-5*` / `o1*` / `o3*` auto-
  dispatch to `/v1/responses` for stateless reasoning round-trip
  (`store=False`, `include=["reasoning.encrypted_content"]`).  Chat
  Completions still serves non-reasoning models.  `use_responses=`
  kwarg forces either path.
- **OpenRouter unified reasoning tokens** — `PyfetchOpenAI` injects a
  provider-native `reasoning` kwarg (Anthropic/Google get `max_tokens`
  since effort→budget conversion is unreliable on those backends;
  everything else gets `effort`) plus a `provider` routing hint
  (`order: ["Anthropic"]` for `anthropic/*` since Bedrock/Vertex
  accept `reasoning` without erroring but silently return no
  reasoning; `require_parameters: True` otherwise).  `tool_choice`
  drops from `required` to `auto` when reasoning is on — Anthropic's
  extended thinking is incompatible with forced tool choice.
  `delta.reasoning_details[]` rides in `ThinkingEmission.signature`
  with the `openrouter-reasoning:` tag prefix and replays byte-for-
  byte on subsequent turns.
- **Anthropic extended thinking** enabled by default via
  `thinking={"type": "enabled", "budget_tokens": 2048}`.
  `_ensure_tool_choice_any` auto-skips tool-force when thinking is
  on (the combo is API-incompatible).
- **Per-emission `emission_id`** on `PrintAction` / `ImageAction`
  parts — the renderer pairs observations back to the specific
  emission that produced them in a multi-emission turn.
- **Forced tool use** — `tool_choice="required"` on OpenAI, `{"type":
  "any"}` on Anthropic (when thinking isn't on),
  `tool_config.mode="ANY"` on Gemini.  Tools are the interface; plain
  assistant text no longer satisfies a turn.
- **Nudges** for missing progress: a `TASK_CONTROL_GUIDANCE`
  OutputEvent when a `python_action` produced no observation (print,
  image, error), and a `NO_PROGRESS_GUIDANCE` SystemNote when a turn
  carried only text/thinking.
- **Streaming file-action content** — `write_file` / `edit_file`
  stream their string args as `file_path` / `file_search` /
  `file_content` `TokenChunk`s so `on_token` handlers can watch file
  writes live.
- **`TextPart` stream event** — all three provider adapters capture
  plain-text content blocks (Anthropic `text`, OpenAI `delta.content`,
  Gemini plain text parts) and surface them as
  `TextEmission`, so user-visible prose mixed into a turn round-
  trips faithfully.
- **`tool_start` marker** — the parser emits a marker on every
  action-tool `ToolCallStart` so the builder preserves
  `PythonEmission` / `TerminalEmission` shape even when `code` /
  `commands` came in empty (no more silent demotion to
  `ThinkingEmission`).
- **Package READMEs** — `agex/llm/formats/tool_use/README.md` covers
  the wire format (emission types, tool surface, `native_thinking`);
  `agex/llm/README.md` covers per-provider reasoning (OpenAI
  Responses, Anthropic extended thinking, Gemini thought parts,
  OpenRouter unified `reasoning_details`) and the signature-packing
  scheme.

### Changed
- **`edit_file` collapsed to search+replace only.**  The
  `insert_after` / `insert_before` operations are gone — insertion is
  expressed as a replace whose `content` includes the anchor.  Models
  were routinely confused by the mutually-exclusive content fields.
- **`pprint_events` / `pprint_tokens`** updated for the emission list:
  terminal commands render, file operations show operation/mode,
  streaming file tokens get emoji prefixes, signed-thinking-only
  turns show an explicit hint, `SystemNoteEvent` bodies are no
  longer silently hidden.
- **Primer** tightened: tools are the interface, text is a side
  channel, observation asymmetry (python/terminal produce
  observations; file tools only confirm writes), append over edit
  for new content.
- **Low-visibility namespaces hidden** from the primer entirely
  instead of emitting a `module <name>:\n    ...` stub.  Saves
  primer tokens and — more importantly — keeps the system_message
  hash stable when helpers auto-register low-viz submodules mid-
  session (previously broke the provider prompt cache at task
  boundaries).  Promotion still works: a low-viz module with a high-
  viz member renders as before.
- **Default reasoning effort** on the OpenAI Responses path bumped
  from `low` to `medium`.
- **`register_io` is idempotent** — marks the agent with
  `_io_registered=True` on first call and no-ops thereafter.  The
  sandbox bridge calls it per `python_action`; the guard keeps the
  agent graph stable across turns.

### Removed
- **XML wire format** — `<THINKING>` / `<CODE>` / `<REPORT>` tag
  parsing and the entire `agex/llm/formats/xml/` directory are gone.
- **`task_continue`** — returning normally from a `python_action` is
  the implicit continue.  No builtin, no exception class.
- **`FileAction` / `EditAction` dataclasses** — superseded by
  `FileWriteEmission` / `FileEditEmission`.
- **`ResponseBuilder` and `check_for_task_call`** — the XML-era
  builder is replaced by `EmissionsBuilder`; terminator detection
  moved into emission handling.
- **Single-`python_action`-per-turn restriction** — multiple Python
  emissions in one turn share a namespace and run in order.
- **`thinking` / `report` schema parameters** (when `native_thinking=True`)
  on `python_action` / `terminal_action` — reasoning and prose ride
  on native content blocks instead.

### Fixed
- Gemini 3 400s (`Function call is missing a thought_signature`) —
  parts are buffered across chunks so late-arriving signatures win;
  a documented dummy signature papers over the first-fc-no-sig edge
  case.
- `PrintAction.emission_id` was always `None` because `execute_sandboxed`
  reset the `_current_emission_id` contextvar before `handle_result`
  read it.  Now passed explicitly.
- OpenAI Chat Completions 400 (`Unrecognized request argument supplied:
  reasoning_effort`) — the Chat path no longer injects
  `reasoning_effort` by default (only reasoning models accept it, and
  they dispatch to Responses).
- `edit_file` failures surface in the `tool_result` instead of being
  masked by the synthesized success line, so the agent can recover
  on the next turn.

## [0.10.2] - 2026-04-20

### Added
- **`FetchAdapter` seam for pyfetch LLM clients**: `PyfetchOpenAI` and `PyfetchAnthropic` accept an optional keyword-only `fetch_adapter` parameter. The adapter owns the HTTP transport; clients handle request construction, SSE parsing, and streaming. Enables custom transports (e.g., routing through a JS bridge so API keys can live outside Python scope). `DefaultPyfetchAdapter` (pyfetch in Pyodide, aiohttp elsewhere) preserves existing behavior when no adapter is passed. See `agex/llm/adapter.py`.
- **Empty-`api_key` support** for pyfetch clients: when `api_key=""` and a `fetch_adapter` is provided, the client omits the `Authorization` / `x-api-key` header, letting the adapter inject auth on the way out. Non-auth headers (e.g., `anthropic-version`) are unaffected.

### Changed
- Transport code (pyfetch / aiohttp fallback) deduplicated from both pyfetch clients into `DefaultPyfetchAdapter`. No behavior change for existing callers.

## [0.10.1] - 2026-04-18

### Fixed
- **Skills not packaged**: `agex/skills/*.md` files were missing from the wheel because `setuptools` only includes Python files by default. Added to `[tool.setuptools.package-data]`.

## [0.10.0] - 2026-04-18

### Added
- **`<REPORT>` tag**: New XML primitive for agent-to-caller communication mid-task. Streams live via `on_token`, persists on `ActionEvent.report`, renders back to the agent's own history for commitment coherence. Sub-agent reports propagate into the parent's observation log via direct `OutputEvent` injection.
- **Python script execution**: Agents can run `python file.py` from `<TERMINAL>` blocks. Scripts execute in a sandtrap sandbox with the agent's full policy but a fresh namespace (no REPL state, no `task_*` bindings). Always available — no opt-in required.
- **Git CLI**: Agents can run `git log`, `git commit -m`, `git diff`, `git branch`, `git checkout`, `git reset --hard`, `git show`, and `git merge` from `<TERMINAL>`, backed by kvgit. History is virtualized — only agent-tagged commits appear; system commits are filtered out. `git reset` restores files without moving kvgit's real HEAD. Opt-in via `register_git(agent)`.
- **`register_git(agent)`**: Registers the git skill and mounts a usage guide at `/skills/git/SKILL.md`.
- **`make_git_handler(vkv, state, vfs)`**: Creates a termish `CommandFunc` for git, suitable for manual wiring into `execute()`.
- **`make_python_handler(agent, fs)`**: Creates a termish `CommandFunc` for python script execution.
- **`build_terminal_commands(agent, fs, state, vfs)`**: Builds the injected commands dict for terminal execution (wired automatically in the agent loop).
- **`pprint_events` / `pprint_tokens`**: Now render `<REPORT>` content — green "Report:" line in events, speech-bubble prefix in token stream.
- **Primer coaching**: `BUILTIN_PRIMER` gains "Communicating with Your Caller" section coaching agents to use `<REPORT>` on multi-turn tasks. Task Control Functions section notes that `task_*` functions are `<PYTHON>`-only.

### Fixed
- **SSE drain**: Flush trailing data on connection close; exit drain early on `message_stop`.

### Changed
- **termish >=0.1.5**: Pluggable command injection via `CommandContext`. All commands unified on a single signature. Parser treats `:@,%+!^` as word characters.

## [0.9.8] - 2026-04-08

### Fixed
- **Unchanged variables no longer re-persisted**: Only reassigned variables are written back to state (identity-based detection). Previously every namespace variable was re-pickled and stored as a new blob on every commit — even if untouched.
- **Multiple EDITs/FILEs to the same file no longer clobber each other**: `ResponseBuilder` now uses unique per-block keys instead of bare file paths.
- **False-positive "already applied" EDITs**: Skips the heuristic when a sibling action in the same batch already modified the file.
- **Duplicate EDIT deduplication**: Identical `<EDIT>` blocks in one response are deduplicated with a warning.
- **Duplicate FILE writes**: Last-write-wins with a warning. Appends preserved.
- **"Already applied" EDIT visibility**: Now emits a `SystemNoteEvent` instead of silently skipping.
- **SSE robustness**: Empty payloads skipped before `json.loads`; incremental UTF-8 decoder flushed at end of stream.

### Added
- **File action confirmations**: Agents receive "✓ Applied file actions: …" after writes/edits.

### Changed
- **kvgit >=0.1.11, termish >=0.1.4**.

## [0.9.7] - 2026-04-05

### Added
- **`PyfetchAnthropic`**: Browser-compatible Anthropic client using `pyodide.http.pyfetch` for direct browser-to-Anthropic API calls without a server proxy. Mirrors `PyfetchOpenAI` but targets the native Anthropic Messages API (with `anthropic-dangerous-direct-browser-access` header, system field, base64 image blocks, and prompt caching). Async-only with SSE streaming.
- **`DEBUG_RAW_STREAM`**: Module-level flag on both pyfetch clients for printing raw SSE text deltas to stdout — useful for debugging model output vs XML-tokenizer behavior.
- **Implicit-close recovery in XML tokenizer**: When an agent forgets to close a section and opens a sibling top-level tag on a new line, the tokenizer transitions cleanly instead of absorbing the new tag into the current section's content. Handles `<TITLE>`, `<THINKING>`, `<PYTHON>`, `<TERMINAL>`, `<FILE>`, and `<EDIT>` as boundaries. Mid-line tag-like strings in file content are preserved.

### Changed
- **No more `<TITLE>` prefill on Anthropic clients**: Removed the assistant-prefill step from both `Anthropic` (SDK-based) and `PyfetchAnthropic`. Letting Claude generate the whole response from scratch improves adherence to the XML format primer — with prefill, the model would occasionally skip `</TITLE>` and `<THINKING>`, then self-correct mid-stream, producing two concatenated attempts that the tokenizer couldn't disambiguate.
- **Stronger XML primer**: Every response must begin with `<TITLE>` + `<THINKING>` (no exceptions, even on continuation turns).

## [0.9.6] - 2026-04-01

### Fixed
- **Instance `network_access` / `host_fs_access`**: These parameters were silently dropped when registering live object instances via `agent.module(obj, name=..., network_access=True)`. Now correctly propagated to the namespace.
- **SuccessEvent rendering restored**: `SuccessEvent` is rendered again (as a user-role observation) so agents can see the repr of programmatically constructed results like DataFrames. `FailEvent` and `ClarifyEvent` remain skipped since they contain literal strings already visible in the preceding code.
- **SuccessEvent token budget**: `render_value` now receives `token_budget` when rendering task results, enabling iterative token-counted DataFrame display (matching how `OutputEvent` already handled it).

### Changed
- **sandtrap >=0.1.11**: Fixes `ContextVar` propagation to `ThreadPoolExecutor` threads, so `network_access=True` works correctly in libraries that use thread pools.

## [0.9.5] - 2026-04-01

### Added
- **Directory-based skills**: `agent.skill()` now accepts directories (Path or importlib Traversable) containing `SKILL.md` and sibling documents. All files are mounted together under `/skills/<name>/`. Dotfiles and dotdirectories are automatically excluded.
- **Image interception**: Base64-encoded images printed via `__AGEX_IMAGE__:` prefix are automatically converted to `ImageAction` objects.
- **Message collapsing**: Consecutive same-role messages in the LLM conversation are merged, producing cleaner conversation histories.

### Fixed
- **Renderer consistency**: `CancelledEvent` and `ClarifyEvent` are now handled identically across markdown and XML renderers. `CancelledEvent` renders as a user message; `ClarifyEvent` is skipped (intent already in preceding code).
- **ChapterEvent role**: Chapters now render as assistant messages (previously user), correctly reflecting that they summarize the agent's prior work.
- **Resilient deserialization**: State decoder catches all unpickling exceptions (not just `RecursionError`), and `get_events_from_log` skips corrupted events instead of crashing.

### Changed
- **Terminal events not rendered**: `FailEvent` and `ClarifyEvent` are no longer sent to the LLM — the agent already expressed its intent via `task_fail()`/`task_clarify()` string literals in the preceding `ActionEvent`.
- **Top-level PIL import**: `PIL.Image`, `base64`, and `io` are now imported at module level in `result.py` with graceful fallback when PIL is unavailable.

## [0.9.4] - 2026-03-19

### Fixed
- **Unpicklable variables surface as errors**: `UnpicklableMarker` now raises a descriptive `UnpicklableVariableError` on any access (attribute, call, comparison, iteration, indexing) instead of silently disappearing from the namespace as a `NameError`. Catches `RecursionError` during unpickling (e.g. pypdf objects).
- **Actionable file edit errors**: When a search/replace edit fails, the error now shows the most similar lines in the file via `difflib.SequenceMatcher`, reports which earlier edits in a batch already succeeded, and skips edits where the replacement is already present.

## [0.9.3] - 2026-03-19

### Changed
- **Compressed image pickling**: `ImageAction` now pickles PIL Images as PNG bytes instead of raw pixel data (~100x smaller storage)
- **Batch event reads**: `get_events_from_log` uses `get_many()` to fetch all events in a single storage transaction instead of N individual reads
- **kvgit >=0.1.8**: Required for fast IndexedDB byte conversion

## [0.9.2] - 2026-03-15

### Added
- **OpenRouter headers**: Support openrouter-friendly headers on the openai compatible endpoint.

### Changed
- **Improved Primer**: Nudge agents to avoid task_fails for recoverable errors

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
