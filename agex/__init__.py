"""agex: agents that work in a versioned world.

agex is a loop over nontainer. A session drives a workspace turn by
turn, and a task runs on a fork of one and hands back a typed value;
both keep their conversation in the workspace's branch, so a checkout
rewinds memory with the files and a fork carries it.

This is the 0.13 rebuild, and the package is empty until its pieces
land. The 0.12 line is at the ``v0.12.4`` tag.
"""

__all__: list[str] = []
