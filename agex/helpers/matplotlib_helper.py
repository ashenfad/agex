"""
Matplotlib registration helpers for agex agents.

Beyond the usual module exposure, this helper handles two
sandbox-environment quirks that matplotlib's lazy initialization
runs into:

1. **Backend selection.**  agex deployments are headless (Pyodide,
   subprocess workers, kernel-isolated containers).  Force the
   non-interactive Agg backend before any pyplot import — backend
   probing for a GUI backend fails in these environments.

2. **Font cache warm-up.**  matplotlib builds its ``fontlist.json``
   cache the first time ``font_manager`` resolves a glyph (i.e. the
   first ``savefig`` with text).  The build acquires a lock file
   inside matplotlib's own package directory.  When the first
   savefig happens *inside* a sandboxed agent turn, the sandbox
   blocks that lock write and savefig fails with
   ``FileNotFoundError`` for a path under ``site-packages``.

   Pre-warming the cache at registration time — while we're still
   at the host level, before any sandboxed code runs — lets the
   lock write succeed and populates the cache for later sandboxed
   savefig calls to read.

``register_matplotlib(agent)`` MUST be called from outside the
sandbox (i.e. before any task execution).  The other helpers in
this package have the same implicit contract; matplotlib is the
first case where violating it has a visible failure mode.
"""

import warnings

from agex.agent import Agent

IO_EXCLUDE = [
    "savefig",
    "imread",
    "imsave",
]

CORE_EXCLUDE = [
    "_*",
    "*._*",
    # Interactive-display calls that are no-ops under Agg but would
    # invite REPL-style usage from agents.
    "show",
    "pause",
    "ion",
    "ioff",
    "isinteractive",
    # Backend switching — pinned to Agg above; agent code shouldn't
    # be able to flip backends mid-session.
    "switch_backend",
]


def _warm_font_cache() -> None:
    """Render a tiny figure with text so matplotlib builds its
    ``fontlist.json`` cache while we're still at the host level.

    Without this, the first sandboxed call to ``Figure.savefig``
    (or any path that needs to render glyphs) triggers the cache
    build, which acquires a lock inside the matplotlib package
    directory — a write the sandbox blocks.
    """
    import io

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        ax.text(0.5, 0.5, "warm")
        fig.savefig(io.BytesIO(), format="png")
    finally:
        plt.close(fig)


def register_matplotlib(agent: Agent, io_friendly: bool = True) -> None:
    """Register matplotlib for agent use.

    Forces the non-interactive Agg backend, pre-warms the font
    cache (best-effort), then exposes the matplotlib namespace
    recursively at low visibility — the API surface is huge and
    agents already know it from training.

    Args:
        agent: The agex agent to register the library on.
        io_friendly: When True (default), keep IO operations like
            ``savefig`` / ``imread`` / ``imsave`` available — the
            sandbox's filesystem layer routes them safely.  Set
            False to block all matplotlib IO at the registration
            boundary.

    Must be called from outside the sandbox; warm-up tries to
    write a font cache file that the sandbox would block.
    """
    try:
        import matplotlib

        # Force Agg before any pyplot import — no agex deployment
        # has a GUI display, and matplotlib's default backend
        # probing fails in headless containers.
        matplotlib.use("Agg")

        try:
            _warm_font_cache()
        except Exception as e:
            warnings.warn(
                f"matplotlib font cache warm-up failed: {e}.  "
                "First savefig() inside a sandboxed turn may fail.",
                UserWarning,
            )

        exclude = CORE_EXCLUDE
        if not io_friendly:
            exclude = exclude + IO_EXCLUDE
        agent.module(
            matplotlib,
            recursive=True,
            visibility="low",
            exclude=exclude,
        )

    except ImportError:
        warnings.warn(
            "matplotlib not installed - skipping matplotlib registration",
            UserWarning,
        )
        raise
