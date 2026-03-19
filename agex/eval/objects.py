"""
Internal representation of objects used by the bridge and render layers.
"""

import io
from dataclasses import dataclass
from typing import Any, Literal

try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None  # type: ignore[assignment, misc]


class PrintAction(tuple):
    """Represents the un-rendered content of a print() call."""

    pass


@dataclass
class ImageAction:
    """Represents an un-rendered image from a view_image() call.

    PIL Images are pickled as compressed PNG bytes to avoid storing
    raw pixel data (~100x smaller).
    """

    image: Any
    detail: Literal["low", "high"] = "high"

    def __getstate__(self) -> dict:
        state = {"detail": self.detail}
        img = self.image
        if _PILImage is not None and isinstance(img, _PILImage.Image):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            state["_png_bytes"] = buf.getvalue()
        else:
            state["image"] = img
        return state

    def __setstate__(self, state: dict) -> None:
        self.detail = state["detail"]
        if "_png_bytes" in state:
            from PIL import Image

            self._png_bytes = state["_png_bytes"]
            self.image = Image.open(io.BytesIO(self._png_bytes))
        else:
            self._png_bytes = None
            self.image = state["image"]

    def png_bytes(self) -> bytes:
        """Return the PNG bytes for this image."""
        cached = getattr(self, "_png_bytes", None)
        if cached is not None:
            return cached
        if _PILImage is None or not isinstance(self.image, _PILImage.Image):
            raise TypeError(
                f"Cannot get PNG bytes for non-PIL image of type "
                f"{type(self.image).__name__}"
            )
        buf = io.BytesIO()
        self.image.save(buf, format="PNG")
        self._png_bytes = buf.getvalue()
        return self._png_bytes

    def _repr_html_(self) -> str:
        """Rich HTML representation for notebook display."""
        # First, try the object's native _repr_html_ method (e.g., plotly figures)
        if hasattr(self.image, "_repr_html_"):
            try:
                return self.image._repr_html_()
            except Exception:
                pass  # Fall through to image serialization

        # For other image types, convert to base64 and display as HTML image
        try:
            # Import here to avoid circular dependency
            from agex.render.primitives import serialize_image_to_base64

            base64_image = serialize_image_to_base64(self.image)
            if base64_image:
                return f'<img src="data:image/png;base64,{base64_image}" style="max-width: 100%; height: auto;" />'
        except Exception:
            pass  # Fall through to text fallback

        # Fallback to text representation
        import html

        type_name = type(self.image).__name__
        escaped_text = html.escape(f"<{type_name} image - display failed>")
        return f'<pre style="background: #f6f8fa; padding: 8px; border-radius: 6px; margin: 0; color: #24292e; font-family: monospace;">{escaped_text}</pre>'
