"""Shared slugify utility for chapter paths."""

import re


def slugify(name: str) -> str:
    """Convert a chapter name to a filesystem-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = slug.strip("-")
    return slug or "unnamed"
