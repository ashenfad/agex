"""
This module provides common registration patterns for popular libraries
to promote safe-by-default agent configurations.
"""

# Recommended exclude pattern for the top-level numpy module.
# This pattern focuses on preventing file I/O and external code execution.
NUMPY_EXCLUDE = [
    # Exclude private/internal members
    "_*",
    # File I/O functions (e.g., save, savetxt, savez)
    "save*",
    # File I/O functions (e.g., load, loadtxt)
    "load*",
    # File I/O functions (e.g., fromfile, tofile)
    "*file",
    "genfromtxt",
    # Memory mapping
    "memmap",
    # F2PY (Fortran to Python interface)
    "f2py",
    # External code interface (ctypes)
    "ctypeslib",
    # The whole testing suite
    "testing",
    # Distutils is deprecated and can be used for system interaction
    "distutils",
]

# Recommended exclude pattern for the top-level pandas module.
# Prevents reading from files/databases and arbitrary code evaluation.
PANDAS_EXCLUDE = [
    # Exclude private/internal members
    "_*",
    # All file/database reading functions (e.g., read_csv, read_sql)
    "read_*",
    # Arbitrary expression evaluation
    "eval",
]

# Recommended exclude pattern for pandas DataFrame/Series objects.
# Prevents writing to files or databases.
DATAFRAME_EXCLUDE = [
    # Exclude private/internal members
    "_*",
    # All file/database writing methods (e.g., to_csv, to_sql)
    "to_*",
]

# Recommended exclude pattern for visualization libraries like matplotlib or seaborn.
# Prevents file I/O and blocking UI operations.
VISUALIZATION_EXCLUDE = [
    # Exclude private/internal members
    "_*",
    # Prevent showing a UI window
    "show",
    # Prevent saving to files (e.g., savefig)
    "save*",
    # Prevent reading from files (e.g., imread)
    "im*",
]

# Recommended exclude pattern for the built-in `random` module.
# Prevents agents from affecting the host's global random state.
RANDOM_EXCLUDE = [
    # Exclude private/internal members
    "_*",
    # State management functions
    "seed",
    "getstate",
    "setstate",
    # Non-deterministic and system-dependent generator
    "SystemRandom",
]
