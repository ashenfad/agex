"""
Helper functions for registering popular libraries with agents.
"""

try:
    from .pandas_helper import register_pandas
except ImportError:
    # pandas not installed
    pass

try:
    from .numpy_helper import register_numpy
except ImportError:
    # numpy not installed
    pass

try:
    from .plotly_helper import register_plotly
except ImportError:
    # plotly not installed
    pass

try:
    from .matplotlib_helper import register_matplotlib
except ImportError:
    # matplotlib not installed
    pass

from .stdlib import register_io, register_stdlib

__all__ = [
    "register_pandas",
    "register_numpy",
    "register_plotly",
    "register_matplotlib",
    "register_stdlib",
    "register_io",
]
