"""
Internal representation of objects used by the bridge and render layers.
"""

import io
from dataclasses import dataclass
from typing import Any, Literal


def _pil_image_cls():
    """Lazily import PIL.Image and return its ``Image`` class (or None).

    PIL loads a non-trivial amount of code at import time; defer it until
    something actually needs to serialize or type-check an image.
    """
    try:
        from PIL import Image  # noqa: PLC0415

        return Image
    except ImportError:
        return None


@dataclass
class PrintAction:
    """Un-rendered content of a ``print()`` call.

    ``args`` holds the positional arguments print() received; the
    renderer joins them with spaces (matching Python's print semantics)
    when producing text for the LLM.

    ``emission_id`` traces this print back to the PythonEmission whose
    execution produced it. Used by the renderer to pair per-emission
    tool_results with the emission that generated them, so a multi-
    action turn's observations don't get mashed together.
    """

    args: tuple = ()
    emission_id: str | None = None

    def __iter__(self):
        return iter(self.args)

    def __len__(self):
        return len(self.args)

    def __getitem__(self, idx):
        return self.args[idx]


@dataclass
class ImageAction:
    """An un-rendered image from a ``view_image()`` call.

    PIL Images are pickled as compressed PNG bytes to avoid storing
    raw pixel data (~100x smaller).

    ``emission_id`` traces this image back to the emission whose
    execution produced it (always a PythonEmission today, since
    ``view_image`` is a sandbox builtin).
    """

    image: Any
    detail: Literal["low", "high"] = "high"
    emission_id: str | None = None

    def __getstate__(self) -> dict:
        state = {"detail": self.detail, "emission_id": self.emission_id}
        img = self.image
        _Image = _pil_image_cls()
        if _Image is not None and isinstance(img, _Image.Image):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            state["_png_bytes"] = buf.getvalue()
        else:
            state["image"] = img
        return state

    def __setstate__(self, state: dict) -> None:
        self.detail = state["detail"]
        self.emission_id = state.get("emission_id")
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
        _Image = _pil_image_cls()
        if _Image is None or not isinstance(self.image, _Image.Image):
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
