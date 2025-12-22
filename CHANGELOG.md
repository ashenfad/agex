# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.7.1]: https://github.com/ashenfad/agex/releases/tag/v0.7.1
[0.7.0]: https://github.com/ashenfad/agex/releases/tag/v0.7.0
[0.6.0]: https://github.com/ashenfad/agex/releases/tag/v0.6.0
[0.5.1]: https://github.com/ashenfad/agex/releases/tag/v0.5.1
[0.5.0]: https://github.com/ashenfad/agex/releases/tag/v0.5.0
