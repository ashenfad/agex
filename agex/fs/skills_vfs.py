"""
Read-only VFS overlay for agent skills.

Builds a virtual /skills directory from skill files registered via agent.skill().
"""

from monkeyfs import ReadOnlyFS, VirtualFS


def create_skills_fs(
    skills: list[tuple[str, bytes]],
) -> ReadOnlyFS | None:
    """Create a read-only VFS from registered skill files.

    Each skill is a (name, content_bytes) tuple. The name is used as
    the directory name under /skills/.

    Args:
        skills: List of (name, content) tuples from agent._skills.

    Returns:
        ReadOnlyFS instance, or None if no skills are registered.
    """
    if not skills:
        return None

    file_dict: dict[str, bytes] = {}
    for name, content in skills:
        path = f"{name}/SKILL.md"
        file_dict[path] = content

    vfs = VirtualFS()
    vfs.write_many(file_dict)
    return ReadOnlyFS(vfs)
