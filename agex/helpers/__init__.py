"""
Helper functions for registering popular libraries with agents.
"""

try:
    from .pandas import register_pandas
except ImportError:
    # pandas not installed
    pass

try:
    from .numpy import register_numpy
except ImportError:
    # numpy not installed
    pass

from .stdlib import register_stdlib
