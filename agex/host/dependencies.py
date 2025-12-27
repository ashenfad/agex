from dataclasses import dataclass, field


@dataclass
class Dependencies:
    """Dependencies required to run an agent."""

    python_version: str
    agex_version: str
    packages: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        """Unique identifier for this set of dependencies (for image caching)."""
        import hashlib

        # Sort packages to ensure stable ID
        payload = f"py{self.python_version}-agex{self.agex_version}-" + "-".join(
            sorted(self.packages)
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]
