"""File editing logic for the task loop.

Handles applying file writes and edits from LLM responses to the filesystem,
with flexible matching strategies for indentation and trailing whitespace.
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from difflib import SequenceMatcher
from typing import Any, Callable

from agex.agent.datatypes import EditAction, FileAction
from agex.agent.events import BaseEvent, SystemNoteEvent
from agex.fs.aware import AgentAwareFS
from agex.llm.core import LLMResponse, ResponseParseError
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

    The algorithm:
    1. Find the first non-empty line in search and strip it
    2. Search for that stripped content in the file
    3. For each potential match, verify all lines match when stripped
    4. Return matches with their positions and the actual matched text

    Args:
        search: The search string from the EDIT action
        content: The file content to search in

    Returns:
        List of (start_pos, end_pos, matched_text) tuples for each match found.
        Positions are byte offsets into content.
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

        # Check if this line matches our anchor (accounting for trailing ws)
        if content_line_stripped != anchor_stripped:
            continue

        # Potential match - calculate where the full block would start
        start_line = i - anchor_idx
        if start_line < 0:
            continue

        # Check if we have enough lines
        end_line = start_line + len(search_lines)
        if end_line > len(content_lines):
            continue

        # Verify all lines match when stripped
        match = True
        for j, search_line_stripped in enumerate(search_stripped):
            content_idx = start_line + j
            content_stripped = content_lines[content_idx].strip()

            # Both should be empty or both should have same stripped content
            # Also handle trailing whitespace flexibility
            if search_line_stripped.rstrip() != content_stripped.rstrip():
                match = False
                break

        if match:
            # Calculate byte positions
            # Sum lengths of all lines before start_line, plus newlines
            start_pos = sum(len(content_lines[k]) + 1 for k in range(start_line))

            # Calculate end position (end of the last matched line)
            matched_lines = content_lines[start_line:end_line]
            matched_text = "\n".join(matched_lines)

            # End position is start + length of matched text
            end_pos = start_pos + len(matched_text)

            matches.append((start_pos, end_pos, matched_text))

    return matches


def _adjust_replacement_indent(replacement: str, search: str, matched_text: str) -> str:
    """Adjust replacement indentation to match the target file's style.

    When we match with flexible indentation, the replacement content needs
    its indentation adjusted to fit naturally into the target file.

    Args:
        replacement: The replacement text from the EDIT action
        search: The original search text (to determine agent's indent baseline)
        matched_text: The actual text that was matched in the file

    Returns:
        Replacement text with indentation adjusted to match the file's style
    """
    search_lines = search.split("\n")
    matched_lines = matched_text.split("\n")
    replacement_lines = replacement.split("\n")

    def get_base_indent_info(lines: list[str]) -> tuple[int, str, int]:
        """Get base indentation info from first non-empty line.

        Returns:
            (indent_in_spaces, indent_char, raw_char_count)
            - indent_in_spaces: equivalent space count (tabs count as 4)
            - indent_char: '\t' if tabs used, ' ' otherwise
            - raw_char_count: actual character count of leading whitespace
        """
        for line in lines:
            stripped = line.lstrip()
            if stripped:
                leading = line[: len(line) - len(stripped)]
                indent_char = "\t" if "\t" in leading else " "
                # Calculate equivalent spaces (tabs = 4 spaces each)
                indent_in_spaces = leading.count("\t") * 4 + leading.count(" ")
                return indent_in_spaces, indent_char, len(leading)
        return 0, " ", 0

    search_base_indent, _, _ = get_base_indent_info(search_lines)
    target_base_indent, target_indent_char, _ = get_base_indent_info(matched_lines)
    replacement_base_indent, _, _ = get_base_indent_info(replacement_lines)

    # Calculate the indent adjustment needed (in equivalent spaces)
    # Heuristic: if replacement base indent matches search base indent,
    # assume agent wrote replacement relative to search context
    if replacement_base_indent == search_base_indent:
        indent_delta = target_base_indent - search_base_indent
    else:
        # Agent used different indent in replacement - shift to match target
        indent_delta = target_base_indent - replacement_base_indent

    # Adjust each line in replacement
    adjusted = []
    for line in replacement_lines:
        stripped = line.lstrip()
        if not stripped:
            # Preserve empty lines (but strip any whitespace for cleanliness)
            adjusted.append("")
        else:
            # Calculate current indent in equivalent spaces
            current_leading = line[: len(line) - len(stripped)]
            current_indent = current_leading.count("\t") * 4 + current_leading.count(
                " "
            )
            new_indent = max(0, current_indent + indent_delta)

            # Use target indent style
            if target_indent_char == "\t":
                # Convert to tabs (4 spaces per tab)
                tabs = new_indent // 4
                spaces = new_indent % 4
                new_leading = "\t" * tabs + " " * spaces
            else:
                new_leading = " " * new_indent

            # Preserve trailing whitespace from original replacement
            trailing = (
                line[len(line) - len(line.rstrip()) :] if line.rstrip() != line else ""
            )
            adjusted.append(new_leading + stripped + trailing)

    return "\n".join(adjusted)


def apply_optimistic_file_actions(
    agent: Any,
    llm_response: LLMResponse,
    fs: Any,
    exec_state: MutableMapping[str, Any],
    on_event: Callable[[BaseEvent], None] | None = None,
) -> None:
    """
    Apply file operations (writes and edits) from the LLM response to the filesystem.

    This is called 'optimistic' because it happens before code execution.
    It allows the agent to import modules it just created.
    """
    if not llm_response.file_actions or not fs:
        return

    # Use underlying FS directly to avoid 'user' source attribution
    target_fs = fs._fs if isinstance(fs, AgentAwareFS) else fs

    # Deduplicate identical EditActions within this response.  Agents
    # occasionally repeat the same edit "defensively" to ensure it applies —
    # but each duplicate re-runs the same search+replace on the previous
    # edit's output, leading to multiplied insertions.  There is no
    # legitimate reason to issue the same EDIT twice in one response
    # (use match_all="true" for multiple matches).
    seen_edits: set[tuple] = set()
    deduped_actions: list[FileAction | EditAction] = []
    dropped_duplicates: list[str] = []
    for action in llm_response.file_actions:
        if isinstance(action, EditAction):
            key = (
                action.path,
                action.search,
                action.content,
                action.operation,
                action.match_all,
            )
            if key in seen_edits:
                dropped_duplicates.append(action.path)
                continue
            seen_edits.add(key)
        deduped_actions.append(action)

    if dropped_duplicates:
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

    # Deduplicate FILE writes to the same path: keep only the last write
    # (agents sometimes emit a first draft then a corrected version in the
    # same response).  Appends are never deduplicated.
    last_write_idx: dict[str, int] = {}
    for i, action in enumerate(deduped_actions):
        if isinstance(action, FileAction) and action.mode != "append":
            last_write_idx[action.path] = i

    dropped_file_writes: list[str] = []
    final_actions: list[FileAction | EditAction] = []
    for i, action in enumerate(deduped_actions):
        if (
            isinstance(action, FileAction)
            and action.mode != "append"
            and last_write_idx.get(action.path) != i
        ):
            dropped_file_writes.append(action.path)
            continue
        final_actions.append(action)

    if dropped_file_writes:
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

    applied: list[str] = []  # track successful actions for error context
    modified_this_batch: set[str] = set()  # files changed by earlier actions

    for action in final_actions:
        if isinstance(action, FileAction):
            path, content, mode = action.path, action.content, action.mode
            fs_mode = "a" if mode == "append" else "w"

            # Check for shadowing
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
            applied.append(f"{'append' if fs_mode == 'a' else 'write'} {path}")
            modified_this_batch.add(path)

        elif isinstance(action, EditAction):
            path = action.path

            # Read existing file
            try:
                existing_content = target_fs.read(path).decode("utf-8")
            except FileNotFoundError:
                raise ResponseParseError(f"File not found for EDIT: {path}")

            # Matching strategy (try in order, stop at first success):
            # 1. Exact match
            # 2. Trailing whitespace flexible match
            # 3. Indent-flexible match (different absolute indentation)

            match_mode = "exact"
            count = existing_content.count(action.search)
            indent_matches: list[tuple[int, int, str]] = []

            if count == 0:
                # Exact match failed - try trailing whitespace flexible matching
                pattern = _build_trailing_ws_pattern(action.search)
                regex_matches = list(pattern.finditer(existing_content))
                count = len(regex_matches)
                match_mode = "trailing_ws"

                if count == 0:
                    # Trailing ws match failed - try indent-flexible matching
                    indent_matches = _find_indent_flexible_match(
                        action.search, existing_content
                    )
                    count = len(indent_matches)
                    match_mode = "indent_flexible"

                    if count == 0:
                        # Check if replacement is already in the file.
                        # Only trust this heuristic if no earlier action in
                        # this batch already modified the file — otherwise
                        # the replacement may appear as a substring of a
                        # sibling edit's content (false positive).
                        if (
                            path not in modified_this_batch
                            and action.content in existing_content
                        ):
                            applied.append(f"edit {path} (already applied)")
                            note = SystemNoteEvent(
                                agent_name="System",
                                message=(
                                    f"⚠️ EDIT {path}: search string not found, "
                                    f"but replacement content appears to already "
                                    f"be in the file — treating as already-applied "
                                    f"and SKIPPING. If this is a false positive "
                                    f"(e.g. your search had a whitespace typo but "
                                    f"the file still needs updating), verify the "
                                    f"file contents and re-issue the edit with a "
                                    f"matching search string."
                                ),
                            )
                            add_event_to_log(exec_state, note, on_event=on_event)
                            continue

                        # Build actionable error with closest match
                        parts = [f"Search string not found in {path}."]

                        similar = _find_similar_lines(action.search, existing_content)
                        if similar:
                            parts.append(
                                "Did you mean to match these lines?\n" + similar
                            )
                        else:
                            search_preview = (
                                action.search[:200] + "..."
                                if len(action.search) > 200
                                else action.search
                            )
                            parts.append(f"Search was:\n{search_preview}")

                        if applied:
                            parts.append(
                                f"Note: {len(applied)} earlier action(s) already "
                                f"applied successfully: {', '.join(applied)}. "
                                f"Do not re-send those."
                            )

                        raise ResponseParseError("\n\n".join(parts))

            if count > 1 and not action.match_all:
                raise ResponseParseError(
                    f"Search string found {count} times in {path}. "
                    f'Use match_all="true" or provide more context.'
                )

            # Apply replacement based on matching mode
            if match_mode == "indent_flexible":
                # Indent-flexible matching - need to adjust replacement indentation
                # Process matches in reverse order to preserve positions
                matches_to_apply = (
                    indent_matches if action.match_all else indent_matches[:1]
                )
                new_content = existing_content

                for start_pos, end_pos, matched_text in reversed(matches_to_apply):
                    # Adjust replacement content's indentation to match the file
                    adjusted_content = _adjust_replacement_indent(
                        action.content, action.search, matched_text
                    )

                    if action.operation == "insert-after":
                        replacement = matched_text + adjusted_content
                    elif action.operation == "insert-before":
                        replacement = adjusted_content + matched_text
                    else:  # "replace"
                        replacement = adjusted_content

                    new_content = (
                        new_content[:start_pos] + replacement + new_content[end_pos:]
                    )

            elif match_mode == "trailing_ws":
                # Trailing whitespace flexible matching - use regex replacement
                pattern = _build_trailing_ws_pattern(action.search)

                def make_replacement(match: re.Match) -> str:
                    matched_text = match.group(0)
                    if action.operation == "insert-after":
                        return matched_text + action.content
                    elif action.operation == "insert-before":
                        return action.content + matched_text
                    else:  # "replace"
                        return action.content

                if action.match_all:
                    new_content = pattern.sub(make_replacement, existing_content)
                else:
                    new_content = pattern.sub(
                        make_replacement, existing_content, count=1
                    )
            else:
                # Exact matching - use str.replace
                if action.operation == "insert-after":
                    replacement = action.search + action.content
                elif action.operation == "insert-before":
                    replacement = action.content + action.search
                else:  # "replace"
                    replacement = action.content

                if action.match_all:
                    new_content = existing_content.replace(action.search, replacement)
                else:
                    new_content = existing_content.replace(
                        action.search, replacement, 1
                    )

            # Write back
            target_fs.write(path, new_content.encode("utf-8"), mode="w")
            applied.append(f"edit {path}")
            modified_this_batch.add(path)

    # Positive confirmation: tell the agent exactly what was applied so it
    # doesn't have to guess whether its file actions worked.  Exclude
    # "already applied" skips — those already got their own warning above.
    confirmed = [a for a in applied if "(already applied)" not in a]
    if confirmed:
        note = SystemNoteEvent(
            agent_name="System",
            message="✓ Applied file actions: " + "; ".join(confirmed),
        )
        add_event_to_log(exec_state, note, on_event=on_event)
