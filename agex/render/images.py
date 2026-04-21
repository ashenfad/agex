"""
Image serialization + cost estimation utilities.

All heavy graphics deps (PIL, matplotlib, plotly) are imported lazily
inside the functions that use them. ``import agex`` stays fast when an
agent doesn't touch images — these modules only get loaded when an
``ImageAction`` actually needs to be rendered or sized.
"""

import base64
import io
from typing import Any


def _is_plotly_figure(image: Any) -> bool:
    """Check if an object is a Plotly figure using duck typing."""
    # Fast duck-typing path — no plotly import needed.
    if hasattr(image, "to_image") and callable(getattr(image, "to_image", None)):
        if hasattr(image, "layout"):
            return True
    # Fallback isinstance check — only triggers the import if the duck
    # test missed and we suspect it might actually be a Plotly figure.
    try:
        import plotly.graph_objects  # noqa: PLC0415
    except ImportError:
        return False
    try:
        return isinstance(image, plotly.graph_objects.Figure)
    except Exception:
        return False


def estimate_image_cost(image: Any, detail: str = "high") -> int:
    """Estimate the token cost for an image.

    Provides a reasonable, model-agnostic estimation for budget management.
    """
    if detail == "low":
        return 85  # Standard low-detail/thumbnail cost.

    width, height = 0, 0

    try:
        from PIL import Image  # noqa: PLC0415

        if isinstance(image, Image.Image):
            width, height = image.size
    except ImportError:
        pass

    if width == 0:
        try:
            import matplotlib.figure  # noqa: PLC0415

            if isinstance(image, matplotlib.figure.Figure):
                dpi = image.get_dpi() or 100.0
                width = int(image.get_figwidth() * dpi)
                height = int(image.get_figheight() * dpi)
        except ImportError:
            pass

    if width == 0 and _is_plotly_figure(image):
        width = image.layout.width or 500
        height = image.layout.height or 400

    if width == 0 or height == 0:
        return 2000

    # Linear pixel-count heuristic — Anthropic's (w*h)/750 is a solid baseline.
    return (width * height) // 750


def serialize_image_to_base64(image: Any) -> str | None:
    """Serialize a supported image type to a PNG base64 string."""
    # Pre-rendered base64 string (e.g. from browser-side Plotly.js)
    if isinstance(image, str):
        return image

    buffer = io.BytesIO()
    try:
        try:
            from PIL import Image  # noqa: PLC0415

            if isinstance(image, Image.Image):
                # Convert to PNG for security + consistency.
                image.save(buffer, format="PNG")
                return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except ImportError:
            pass

        try:
            import matplotlib.figure  # noqa: PLC0415

            if isinstance(image, matplotlib.figure.Figure):
                image.savefig(buffer, format="png", bbox_inches="tight")
                return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except ImportError:
            pass

        if _is_plotly_figure(image):
            # kaleido is used by plotly to export static images.
            if hasattr(image, "to_image") and callable(
                getattr(image, "to_image", None)
            ):
                image_bytes = image.to_image(format="png")
                return base64.b64encode(image_bytes).decode("utf-8")
    except Exception:
        # Caller generates appropriate error messages on failure.
        return None

    return None


def get_image_error_message(image: Any) -> str:
    """Generate a helpful error message for failed image serialization."""
    if not _is_plotly_figure(image):
        return f"<unsupported image type: {type(image).__name__}>"

    # Try to get the actual error from Plotly export.
    error_msg = None
    try:
        if hasattr(image, "to_image") and callable(getattr(image, "to_image", None)):
            image.to_image(format="png")
    except Exception as e:
        error_msg = str(e)

    if error_msg and ("kaleido" in error_msg.lower()):
        return (
            "<Plotly figure export failed: Kaleido package is required. "
            "Install with: pip install kaleido>"
        )
    elif error_msg:
        return f"<Plotly figure export failed: {error_msg}>"
    else:
        return (
            "<Plotly figure export failed: Kaleido package may be missing. "
            "Install with: pip install kaleido>"
        )
