"""
Standard library registration helpers for agex agents.

This module provides helper functions to register useful Python standard
library modules with agents, focusing on safe mathematical, utility, and
data processing modules.
"""

import base64
import calendar
import collections
import csv
import datetime
import decimal
import fractions
import hashlib
import io
import itertools
import json
import math
import os
import pathlib
import pickle
import random
import re
import statistics
import string
import textwrap
import time
import traceback
import typing
import uuid
import zoneinfo

from agex.agent import Agent

# Exclude global state functions from random module
RANDOM_EXCLUDE = [
    "_*",
    "seed",
    "getstate",
    "setstate",
    "SystemRandom",
]


def register_io(agent: Agent) -> None:
    """Register IO related modules with the agent for VFS operations.

    Idempotent — subsequent calls on the same agent are no-ops.  This
    matters because registering modules mutates the agent's system
    message (via ``render_definitions``), so re-registering mid-session
    after the first ``_build_system_message`` has run would flip the
    provider prompt cache on the next task (sys_hash would drift).
    """
    if getattr(agent, "_io_registered", False):
        return
    agent._io_registered = True  # type: ignore[attr-defined]
    # File-like objects needed by VFS and IsolatedFS
    agent.module(io, visibility="low", include=["BytesIO", "StringIO", "TextIOWrapper"])

    # Register actual file types from _io module (C implementation)
    # These are the real types returned by builtin open() in isolated FS
    import _io

    agent.cls(_io.TextIOWrapper, visibility="low")
    agent.cls(_io.BufferedReader, visibility="low")
    agent.cls(_io.BufferedWriter, visibility="low")
    agent.cls(_io.BufferedRandom, visibility="low")

    # Register os.stat_result so attributes like st_size are accessible
    agent.cls(os.stat_result, visibility="low")

    # File system operations (VFS-aware wrappers exist for these)
    # Note: os.path is a submodule, need to register separately
    agent.module(
        os,
        visibility="low",
        include=[
            "listdir",
            "remove",
            "unlink",
            "mkdir",
            "makedirs",
            "rename",
            "stat",
            "getcwd",
            "chdir",
        ],
    )
    agent.module(
        os.path,
        visibility="low",
        include=[
            "exists",
            "isfile",
            "isdir",
            "islink",
            "lexists",
            "samefile",
            "realpath",
            "join",
            "basename",
            "dirname",
            "splitext",
        ],
    )

    # Serialization formats for file content
    agent.module(json, visibility="low")
    agent.module(csv, visibility="low")
    agent.module(pickle, visibility="low")

    # pathlib - exclude methods that bypass VFS wrappers
    # Agents must use open() instead of Path.read_text()  etc.
    agent.module(
        pathlib,
        visibility="low",
        exclude=["Path.open", "Path.read_*", "Path.write_*"],
    )

    agent.fn(open, visibility="low")


def register_stdlib(agent: Agent, io_friendly: bool = True) -> None:
    """Register useful Python standard library modules with the agent."""

    # Mathematical modules
    agent.module(math, visibility="low")
    agent.module(random, visibility="low", exclude=RANDOM_EXCLUDE)
    agent.module(statistics, visibility="low")
    agent.module(decimal, visibility="low")
    agent.module(fractions, visibility="low")
    agent.module(time, visibility="low")

    # Utility modules
    agent.module(collections, visibility="low")
    agent.module(itertools, visibility="low")

    # Date/time modules
    agent.module(calendar, visibility="low")
    agent.module(datetime, visibility="low")
    agent.cls(datetime.datetime, visibility="low")
    agent.cls(datetime.date, visibility="low")
    agent.cls(datetime.time, visibility="low")
    agent.cls(datetime.timedelta, visibility="low")
    agent.cls(datetime.timezone, visibility="low")
    agent.cls(datetime.tzinfo, visibility="low")

    # String and text processing
    agent.module(re, visibility="low")
    agent.module(string, visibility="low")
    agent.module(textwrap, visibility="low")

    # Debugging (safe formatting functions only)
    agent.module(
        traceback,
        visibility="low",
        include=["format_exc", "format_exception", "print_exc"],
    )

    # Data encoding/processing (json/csv registered by register_io when io_friendly)
    agent.module(base64, visibility="low")
    agent.module(uuid, visibility="low")
    agent.module(hashlib, visibility="low")
    agent.module(zoneinfo, visibility="low")

    # IO registration
    if io_friendly:
        register_io(agent)
    else:
        agent.module(
            io, visibility="low", include=["BytesIO", "StringIO", "TextIOWrapper"]
        )
    # Exclude deprecated typing.io and typing.re (removed in Python 3.13)
    agent.module(typing, visibility="low", exclude=["io", "re"])
