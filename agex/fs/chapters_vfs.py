"""
Read-only VFS overlay for browsing chaptered event history.

Builds a virtual directory tree from ChapterEvents, allowing agents
to browse chaptered history using existing file tools (read, grep, ls).
"""

from monkeyfs import ReadOnlyFS, VirtualFS

from agex.agent.events import BaseEvent, ChapterEvent
from agex.fs.slugify import slugify as _slugify


def _event_type_label(event: BaseEvent) -> str:
    """Get a short label for an event type."""
    class_name = type(event).__name__
    # Remove 'Event' suffix for brevity
    if class_name.endswith("Event"):
        class_name = class_name[:-5]
    return class_name.lower()


def _unique_slug(base_slug: str, path_prefix: str, file_dict: dict[str, bytes]) -> str:
    """Return a slug that doesn't collide with existing entries in file_dict."""
    candidate = base_slug
    counter = 2
    while f"{path_prefix}/{candidate}/summary.md" in file_dict:
        candidate = f"{base_slug}-{counter}"
        counter += 1
    return candidate


def _build_chapter_entries(
    events: list[BaseEvent],
    path_prefix: str,
    file_dict: dict[str, bytes],
) -> None:
    """Recursively build VFS entries for a list of chaptered events."""
    for i, event in enumerate(events, 1):
        if isinstance(event, ChapterEvent):
            slug = _unique_slug(_slugify(event.name), path_prefix, file_dict)
            chapter_path = f"{path_prefix}/{slug}"

            # summary.md
            summary = f"# {event.name}\n\n{event.message}\n"
            file_dict[f"{chapter_path}/summary.md"] = summary.encode("utf-8")

            # Nested events
            if event.events:
                for j, nested in enumerate(event.events, 1):
                    if isinstance(nested, ChapterEvent):
                        # Recurse for nested chapters
                        _build_chapter_entries(
                            [nested],
                            f"{chapter_path}/chapters",
                            file_dict,
                        )
                    else:
                        label = _event_type_label(nested)
                        event_path = f"{chapter_path}/events/{j:03d}-{label}.md"
                        content = nested._repr_markdown_()
                        file_dict[event_path] = content.encode("utf-8")


def build_chapters_dict(events: list[BaseEvent]) -> dict[str, bytes]:
    """Build a dict of path -> content for all ChapterEvents.

    Args:
        events: List of events from the log (top level).

    Returns:
        Dict mapping VFS paths to file content bytes.
    """
    file_dict: dict[str, bytes] = {}

    chapter_events = [e for e in events if isinstance(e, ChapterEvent)]
    if not chapter_events:
        return file_dict

    _build_chapter_entries(chapter_events, "", file_dict)
    return file_dict


def create_chapters_fs(
    events: list[BaseEvent],
) -> ReadOnlyFS | None:
    """Create a read-only VFS from ChapterEvents.

    Args:
        events: List of events from the log.

    Returns:
        ReadOnlyFS instance, or None if no ChapterEvents exist.
    """
    file_dict = build_chapters_dict(events)
    if not file_dict:
        return None

    vfs = VirtualFS()
    vfs.write_many(file_dict)
    return ReadOnlyFS(vfs)
