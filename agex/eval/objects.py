"""
Internal representation of objects used by the bridge and render layers.
"""

from dataclasses import dataclass
from typing import Any, Literal


class PrintAction(tuple):
    """Represents the un-rendered content of a print() call."""

    pass


@dataclass
class ImageAction:
    """Represents an un-rendered image from a view_image() call."""

    image: Any
    detail: Literal["low", "high"] = "high"

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
            from agex.render.stream import _serialize_image_to_base64

            base64_image = _serialize_image_to_base64(self.image)
            if base64_image:
                return f'<img src="data:image/png;base64,{base64_image}" style="max-width: 100%; height: auto;" />'
        except Exception:
            pass  # Fall through to text fallback

        # Fallback to text representation
        import html

        type_name = type(self.image).__name__
        escaped_text = html.escape(f"<{type_name} image - display failed>")
        return f'<pre style="background: #f6f8fa; padding: 8px; border-radius: 6px; margin: 0; color: #24292e; font-family: monospace;">{escaped_text}</pre>'
