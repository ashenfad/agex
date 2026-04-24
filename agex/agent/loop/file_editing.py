"""File editing logic for the task loop.

Applies :class:`FileWriteEmission` / :class:`FileEditEmission` objects
to the filesystem, with flexible matching strategies for indentation
and trailing whitespace.
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from difflib import SequenceMatcher
from typing import Any, Callable

from agex.agent.emissions import FileEditEmission, FileWriteEmission
from agex.agent.events import BaseEvent, SystemNoteEvent
from agex.fs.aware import AgentAwareFS
from agex.llm.core import ResponseParseError
from agex.state.log import add_event_to_log


def _find_similar_lines(
    search: str, content: str, threshold: float = 0.6, context: int = 3
) -> str | None:
    """Find the most similar chunk in content and return it with context.

    Returns a formatted string showing the best match, or None if nothing
    is similar enough.
    """
    search_lines = search.splitlines()
    content_lines = content.splitlines()
    n = len(search_lines)

    if not search_lines or not content_lines:
        return None

    best_ratio = 0.0
    best_start = 0

    for i in range(len(content_lines) - n + 1):
        candidate = "\n".join(content_lines[i : i + n])
        ratio = SequenceMatcher(None, search, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i

    if best_ratio < threshold:
        return None

    # Show the best match with context lines
    ctx_start = max(0, best_start - context)
    ctx_end = min(len(content_lines), best_start + n + context)
    snippet_lines = []
    for i in range(ctx_start, ctx_end):
        marker = ">" if best_start <= i < best_start + n else " "
        snippet_lines.append(f"{marker} {i + 1:4d} | {content_lines[i]}")

    return "\n".join(snippet_lines)


def _build_trailing_ws_pattern(search: str) -> re.Pattern:
    """Build a regex pattern that matches search with flexible trailing whitespace.

    This allows the search to match even if the file has trailing spaces/tabs
    at the end of lines that the agent didn't include in the search string.
    Internal whitespace (indentation) is preserved exactly.
    """
    lines = search.split("\n")
    pattern_parts = []
    for line in lines:
        # Escape the line for regex, strip trailing whitespace
        escaped = re.escape(line.rstrip())
        # Allow optional trailing whitespace (spaces/tabs only, not newlines)
        pattern_parts.append(escaped + r"[ \t]*")

    # Join with literal newline
    pattern = "\n".join(pattern_parts)
    return re.compile(pattern)


def _find_indent_flexible_match(
    search: str, content: str
) -> list[tuple[int, int, str]]:
    """Find matches for search in content with flexible indentation.

    This handles cases where the search and content have the same code structure
    but different absolute indentation levels (e.g., agent sends 2-space indent
    but file uses 4-space or tabs).
    """
    search_lines = search.split("\n")
    content_lines = content.split("\n")

    # Find first non-empty line in search for anchoring
    anchor_stripped = None
    anchor_idx = 0
    for idx, line in enumerate(search_lines):
        stripped = line.strip()
        if stripped:
            anchor_stripped = stripped
            anchor_idx = idx
            break

    if anchor_stripped is None:
        return []

    # Build list of stripped search lines for comparison
    search_stripped = [line.strip() for line in search_lines]

    matches = []

    # Search for anchor line in content
    for i, content_line in enumerate(content_lines):
        content_line_stripped = content_line.strip()

        if content_line_stripped != anchor_stripped:
            continue

        start_line = i - anchor_idx
        if start_line < 0:
            continue

        end_line = start_line + len(search_lines)
        if end_line > len(content_lines):
            continue

        match = True
        for j, search_line_stripped in enumerate(search_stripped):
            content_idx = start_line + j
            content_stripped = content_lines[content_idx].strip()

            if search_line_stripped.rstrip() != content_stripped.rstrip():
                match = False
                break

        if match:
            start_pos = sum(len(content_lines[k]) + 1 for k in range(start_line))

            matched_lines = content_lines[start_line:end_line]
            matched_text = "\n".join(matched_lines)

            end_pos = start_pos + len(matched_text)

            matches.append((start_pos, end_pos, matched_text))

    return matches


def _adjust_replacement_indent(replacement: str, search: str, matched_text: str) -> str:
    """Adjust replacement indentation to match the target file's style."""
    search_lines = search.split("\n")
    matched_lines = matched_text.split("\n")
    replacement_lines = replacement.split("\n")

    def get_base_indent_info(lines: list[str]) -> tuple[int, str, int]:
        for line in lines:
            stripped = line.lstrip()
            if stripped:
                leading = line[: len(line) - len(stripped)]
                indent_char = "\t" if "\t" in leading else " "
                indent_in_spaces = leading.count("\t") * 4 + leading.count(" ")
                return indent_in_spaces, indent_char, len(leading)
        return 0, " ", 0

    search_base_indent, _, _ = get_base_indent_info(search_lines)
    target_base_indent, target_indent_char, _ = get_base_indent_info(matched_lines)
    replacement_base_indent, _, _ = get_base_indent_info(replacement_lines)

    if replacement_base_indent == search_base_indent:
        indent_delta = target_base_indent - search_base_indent
    else:
        indent_delta = target_base_indent - replacement_base_indent

    adjusted = []
    for line in replacement_lines:
        stripped = line.lstrip()
        if not stripped:
            adjusted.append("")
        else:
            current_leading = line[: len(line) - len(stripped)]
            current_indent = current_leading.count("\t") * 4 + current_leading.count(
                " "
            )
            new_indent = max(0, current_indent + indent_delta)

            if target_indent_char == "\t":
                tabs = new_indent // 4
                spaces = new_indent % 4
                new_leading = "\t" * tabs + " " * spaces
            else:
                new_leading = " " * new_indent

            trailing = (
                line[len(line) - len(line.rstrip()) :] if line.rstrip() != line else ""
            )
            adjusted.append(new_leading + stripped + trailing)

    return "\n".join(adjusted)


def apply_optimistic_file_actions(
    agent: Any,
    llm_response: Any,
    fs: Any,
    exec_state: MutableMapping[str, Any],
    on_event: Callable[[BaseEvent], None] | None = None,
) -> None:
    """Legacy convenience: walk ``llm_response.file_actions`` and apply
    each to the filesystem.

    Retains the original dedup heuristics: identical EDIT emissions are
    collapsed to one (with a warning) so agents that re-issue the same
    edit "defensively" don't multiply the replacement.  Consecutive
    WRITE emissions to the same path keep only the last (agents
    sometimes emit a first draft then a corrected version).  Append
    writes are never deduplicated.

    New code should walk emissions directly via
    :func:`apply_file_write` / :func:`apply_file_edit` — the retooling
    loop does this per-emission, interleaving with Python execution so
    ``write_file → python import`` works on the same turn.  This helper
    is here for Phase 2 test-fixture compatibility.
    """
    # Accept an LLMResponse/ActionEvent (walk its ``emissions``), a raw
    # list of emissions, or any ducktyped object with ``file_actions``.
    if hasattr(llm_response, "emissions"):
        actions = [
            em
            for em in llm_response.emissions
            if isinstance(em, (FileWriteEmission, FileEditEmission))
        ]
    elif isinstance(llm_response, (list, tuple)):
        actions = list(llm_response)
    else:
        actions = list(getattr(llm_response, "file_actions", None) or [])
    if not actions or not fs:
        return

    seen_edits: set[tuple] = set()
    deduped: list[Any] = []
    dropped_duplicates: list[str] = []
    for action in actions:
        if isinstance(action, FileEditEmission):
            key = (
                action.path,
                action.search,
                action.content,
                action.match_all,
            )
            if key in seen_edits:
                dropped_duplicates.append(action.path)
                continue
            seen_edits.add(key)
        deduped.append(action)

    if dropped_duplicates:
        from agex.agent.events import SystemNoteEvent

        note = SystemNoteEvent(
            agent_name="System",
            message=(
                f"⚠️ Dropped {len(dropped_duplicates)} duplicate EDIT block(s) "
                f"targeting: {', '.join(sorted(set(dropped_duplicates)))}. "
                f"Each unique EDIT runs once — do not repeat the same "
                f'search+replace defensively.  Use match_all="true" if you '
                f"need a pattern applied to multiple matches."
            ),
        )
        add_event_to_log(exec_state, note, on_event=on_event)

    last_write_idx: dict[str, int] = {}
    for i, action in enumerate(deduped):
        if isinstance(action, FileWriteEmission) and action.mode != "append":
            last_write_idx[action.path] = i

    dropped_file_writes: list[str] = []
    final_actions: list[Any] = []
    for i, action in enumerate(deduped):
        if (
            isinstance(action, FileWriteEmission)
            and action.mode != "append"
            and last_write_idx.get(action.path) != i
        ):
            dropped_file_writes.append(action.path)
            continue
        final_actions.append(action)

    if dropped_file_writes:
        from agex.agent.events import SystemNoteEvent

        note = SystemNoteEvent(
            agent_name="System",
            message=(
                f"⚠️ Dropped {len(dropped_file_writes)} earlier FILE write(s) "
                f"superseded by a later write to the same path: "
                f"{', '.join(sorted(set(dropped_file_writes)))}. "
                f"Only the last <FILE> per path is applied."
            ),
        )
        add_event_to_log(exec_state, note, on_event=on_event)

    applied: list[str] = []
    modified_this_batch: set[str] = set()

    for action in final_actions:
        if isinstance(action, FileWriteEmission):
            apply_file_write(agent, action, fs, exec_state, on_event=on_event)
            verb = "append" if action.mode == "append" else "write"
            applied.append(f"{verb} {action.path}")
            modified_this_batch.add(action.path)
        elif isinstance(action, FileEditEmission):
            if apply_file_edit(
                action,
                fs,
                exec_state,
                on_event=on_event,
                modified_this_batch=modified_this_batch,
            ):
                applied.append(f"edit {action.path}")
                modified_this_batch.add(action.path)

    if applied:
        from agex.agent.events import SystemNoteEvent

        note = SystemNoteEvent(
            agent_name="System",
            message="✓ Applied file actions: " + "; ".join(applied),
        )
        add_event_to_log(exec_state, note, on_event=on_event)


def apply_file_write(
    agent: Any,
    emission: FileWriteEmission,
    fs: Any,
    exec_state: MutableMapping[str, Any],
    on_event: Callable[[BaseEvent], None] | None = None,
) -> None:
    """Apply a single :class:`FileWriteEmission` to the filesystem."""
    if not fs:
        return
    target_fs = fs._fs if isinstance(fs, AgentAwareFS) else fs
    path, content, mode = emission.path, emission.content, emission.mode
    fs_mode = "a" if mode == "append" else "w"

    # Warn if this would shadow a registered Python module so the agent
    # doesn't wonder why its file wasn't picked up by import.
    if path.endswith(".py"):
        module_name = path[:-3].replace("/", ".")
        if module_name in agent._policy.namespaces:
            warning = SystemNoteEvent(
                agent_name="System",
                message=(
                    f"⚠️ Warning: Created file '{path}' shadows registered system "
                    f"module '{module_name}'. The system module will take precedence "
                    f"during imports."
                ),
            )
            add_event_to_log(exec_state, warning, on_event=on_event)

    target_fs.write(path, content.encode("utf-8"), mode=fs_mode)


def apply_file_edit(
    emission: FileEditEmission,
    fs: Any,
    exec_state: MutableMapping[str, Any],
    on_event: Callable[[BaseEvent], None] | None = None,
    modified_this_batch: set[str] | None = None,
) -> bool:
    """Apply a single :class:`FileEditEmission` to the filesystem.

    Returns ``True`` when the file was actually modified, ``False`` when
    the replacement content was already present (the "already applied"
    no-op path).  Raises :class:`ResponseParseError` with actionable
    context if the search string can't be located and isn't already in
    the file.

    ``modified_this_batch`` gates the "already applied" heuristic:
    when the caller knows the file was touched by an earlier action in
    the current batch, the heuristic is suppressed (otherwise a write
    immediately followed by an edit whose replacement happens to match
    the newly-written content would falsely look like an idempotent
    re-edit).
    """
    if not fs:
        return False
    modified = modified_this_batch or set()
    target_fs = fs._fs if isinstance(fs, AgentAwareFS) else fs
    path = emission.path

    try:
        existing_content = target_fs.read(path).decode("utf-8")
    except FileNotFoundError:
        raise ResponseParseError(f"File not found for EDIT: {path}")

    # Matching strategy (try in order, stop at first success):
    # 1. Exact match
    # 2. Trailing whitespace flexible match
    # 3. Indent-flexible match (different absolute indentation)

    match_mode = "exact"
    count = existing_content.count(emission.search)
    indent_matches: list[tuple[int, int, str]] = []

    if count == 0:
        pattern = _build_trailing_ws_pattern(emission.search)
        regex_matches = list(pattern.finditer(existing_content))
        count = len(regex_matches)
        match_mode = "trailing_ws"

        if count == 0:
            indent_matches = _find_indent_flexible_match(
                emission.search, existing_content
            )
            count = len(indent_matches)
            match_mode = "indent_flexible"

            if count == 0:
                # Treat already-applied replacements as a no-op so the
                # agent's accidental retries don't error.  Skip this
                # shortcut if an earlier action in the same batch just
                # modified the file — otherwise a write followed by an
                # edit whose replacement happens to match the newly-
                # written content trips a false positive.
                if path not in modified and emission.content in existing_content:
                    note = SystemNoteEvent(
                        agent_name="System",
                        message=(
                            f"⚠️ EDIT {path}: search string not found, "
                            f"but replacement content appears to already "
                            f"be in the file — treating as already-applied "
                            f"and SKIPPING."
                        ),
                    )
                    add_event_to_log(exec_state, note, on_event=on_event)
                    return False

                parts = [f"Search string not found in {path}."]
                similar = _find_similar_lines(emission.search, existing_content)
                if similar:
                    parts.append("Did you mean to match these lines?\n" + similar)
                else:
                    search_preview = (
                        emission.search[:200] + "..."
                        if len(emission.search) > 200
                        else emission.search
                    )
                    parts.append(f"Search was:\n{search_preview}")
                # Include the batch context so the agent can see which
                # edits landed before this one blew up.
                prior = sorted(modified)
                if prior:
                    parts.append(
                        "Note: earlier action(s) in this batch already "
                        "modified: " + ", ".join(prior) + ". Do not re-send those."
                    )
                raise ResponseParseError("\n\n".join(parts))

    if count > 1 and not emission.match_all:
        raise ResponseParseError(
            f"Search string found {count} times in {path}. "
            f'Use match_all="true" or provide more context.'
        )

    if match_mode == "indent_flexible":
        matches_to_apply = indent_matches if emission.match_all else indent_matches[:1]
        new_content = existing_content

        for start_pos, end_pos, matched_text in reversed(matches_to_apply):
            replacement = _adjust_replacement_indent(
                emission.content, emission.search, matched_text
            )
            new_content = new_content[:start_pos] + replacement + new_content[end_pos:]

    elif match_mode == "trailing_ws":
        pattern = _build_trailing_ws_pattern(emission.search)
        if emission.match_all:
            new_content = pattern.sub(emission.content, existing_content)
        else:
            new_content = pattern.sub(emission.content, existing_content, count=1)
    else:
        # Exact matching
        if emission.match_all:
            new_content = existing_content.replace(emission.search, emission.content)
        else:
            new_content = existing_content.replace(emission.search, emission.content, 1)

    target_fs.write(path, new_content.encode("utf-8"), mode="w")
    return True
