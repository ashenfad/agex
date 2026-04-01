"""
Read-only VFS overlay for agent skills.

Builds a virtual /skills directory from skill files registered via agent.skill().
"""

from monkeyfs import ReadOnlyFS, VirtualFS


def create_skills_fs(
    skills: list[tuple[str, dict[str, bytes]]],
) -> ReadOnlyFS | None:
    """Create a read-only VFS from registered skill files.

    Each skill is a (name, files_dict) tuple where files_dict maps relative
    paths to file contents. The name is used as the directory name under /skills/.

    Args:
        skills: List of (name, files_dict) tuples from agent._skills.

    Returns:
        ReadOnlyFS instance, or None if no skills are registered.
    """
    if not skills:
        return None

    file_dict: dict[str, bytes] = {}
    for name, files in skills:
        for rel_path, content in files.items():
            file_dict[f"{name}/{rel_path}"] = content

    vfs = VirtualFS()
    vfs.write_many(file_dict)
    return ReadOnlyFS(vfs)
