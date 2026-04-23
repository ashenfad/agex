"""Concise builders for emission-based test fixtures.

Tests construct :class:`LLMResponse` objects and emission lists
frequently — often tens of times per file.  These helpers keep the
call sites readable while making it trivial to skim what each turn
contains.

Use from a test module with a terse import alias:

    from tests.agex._emissions import emit as e

    LLMResponse(emissions=[e.think("plan"), e.py("task_success(1)")])

Or import the specific helpers directly.
"""

from agex.agent.emissions import (
    FileEditEmission,
    FileWriteEmission,
    PythonEmission,
    TerminalEmission,
    TextEmission,
    ThinkingEmission,
)
from agex.llm.core import LLMResponse


def py(
    code: str, *, thinking: str | None = None, title: str | None = None
) -> PythonEmission:
    return PythonEmission(code=code, thinking=thinking, title=title)


def term(
    commands: str, *, thinking: str | None = None, title: str | None = None
) -> TerminalEmission:
    return TerminalEmission(commands=commands, thinking=thinking, title=title)


def write(path: str, content: str, *, mode: str = "write") -> FileWriteEmission:
    return FileWriteEmission(path=path, content=content, mode=mode)  # type: ignore[arg-type]


def edit(
    path: str,
    search: str,
    content: str,
    *,
    operation: str = "replace",
    match_all: bool = False,
) -> FileEditEmission:
    return FileEditEmission(
        path=path,
        search=search,
        content=content,
        operation=operation,  # type: ignore[arg-type]
        match_all=match_all,
    )


def text(value: str) -> TextEmission:
    return TextEmission(text=value)


def think(value: str) -> ThinkingEmission:
    return ThinkingEmission(text=value)


def response(*emissions) -> LLMResponse:
    """Build an :class:`LLMResponse` from positional emissions."""
    return LLMResponse(emissions=list(emissions))


def py_response(
    code: str,
    *,
    thinking: str | None = None,
    title: str | None = None,
) -> LLMResponse:
    """Shorthand: one PythonEmission wrapped in an LLMResponse.

    Covers the single most common fixture pattern.
    """
    return LLMResponse(
        emissions=[PythonEmission(code=code, thinking=thinking, title=title)]
    )


def make_response(
    thinking: str = "",
    code: str | None = None,
    *,
    title: str = "",
    report: str = "",
    file_actions: list | None = None,
    terminal: str | None = None,
    emissions: list | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> LLMResponse:
    """Build an :class:`LLMResponse` from pre-retooling kwargs.

    Test-only convenience that mirrors the shape old fixtures used:

        make_response(thinking="...", code="...", file_actions=[...])

    The legacy fields are translated into an emission list:

    - ``file_actions`` (accepts dicts or :class:`FileWriteEmission` /
      :class:`FileEditEmission` instances) come first, each as its own
      emission.
    - ``thinking`` + ``code`` → :class:`PythonEmission(code, thinking)`.
    - ``thinking`` + ``terminal`` → :class:`TerminalEmission(commands, thinking)`.
    - ``thinking`` alone → :class:`ThinkingEmission`.
    - ``report`` appends a :class:`TextEmission`.

    Exists only so tests that were written against the pre-retooling
    response shape stay readable; new tests should build emissions
    directly via :func:`py`, :func:`write`, etc.

    Callers that already have an emission list can pass it via
    ``emissions=`` — it short-circuits straight to
    :class:`LLMResponse` so ``make_response`` is a drop-in rename for
    both old- and new-shape fixtures.
    """
    if emissions is not None:
        return LLMResponse(
            emissions=list(emissions),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    emissions = []

    for fa in file_actions or []:
        emissions.append(_coerce_file_action(fa))

    title_or_none = title or None
    thinking_or_none = thinking or None

    if terminal is not None:
        emissions.append(
            TerminalEmission(
                commands=terminal, title=title_or_none, thinking=thinking_or_none
            )
        )
    elif code is not None:
        emissions.append(
            PythonEmission(code=code, title=title_or_none, thinking=thinking_or_none)
        )
    elif thinking:
        emissions.append(ThinkingEmission(text=thinking))

    if report:
        emissions.append(TextEmission(text=report))

    return LLMResponse(
        emissions=emissions,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _coerce_file_action(fa):
    """Normalize a legacy-shape file-action fixture into its emission."""
    if isinstance(fa, (FileWriteEmission, FileEditEmission)):
        return fa

    if isinstance(fa, dict):
        if "search" in fa:
            return FileEditEmission(
                path=fa.get("path", ""),
                search=fa.get("search", ""),
                content=fa.get(
                    "content",
                    fa.get("replace")
                    or fa.get("insert_after")
                    or fa.get("insert_before")
                    or "",
                ),
                operation=fa.get("operation", "replace"),
                match_all=bool(fa.get("match_all", False)),
            )
        return FileWriteEmission(
            path=fa.get("path", ""),
            content=fa.get("content", ""),
            mode=fa.get("mode", "write"),
        )

    raise TypeError(f"Cannot coerce {type(fa).__name__} into a file emission")


# --- Readers for test assertions ------------------------------------------
# Tests frequently want "what's the code for this event" or "what files
# did this turn write" — these helpers keep the assertion site brief
# without propping up the old named-field shape in production code.


def event_code(event) -> str | None:
    """Concatenated code across this ActionEvent's PythonEmissions,
    or ``None`` if the turn had no Python."""
    pieces = [em.code for em in event.emissions if isinstance(em, PythonEmission)]
    if not pieces:
        return None
    return "\n".join(pieces)


def event_thinking(event) -> str:
    """Concatenated narration across this ActionEvent's emissions."""
    pieces: list[str] = []
    for em in event.emissions:
        if isinstance(em, (PythonEmission, TerminalEmission)) and em.thinking:
            pieces.append(em.thinking)
        elif isinstance(em, ThinkingEmission) and em.text and not em.redacted:
            pieces.append(em.text)
    return "\n".join(pieces)


def event_title(event) -> str:
    for em in event.emissions:
        title = getattr(em, "title", None)
        if title:
            return title
    return ""


def event_terminal(event) -> str | None:
    pieces = [em.commands for em in event.emissions if isinstance(em, TerminalEmission)]
    if not pieces:
        return None
    return "\n".join(pieces)


def event_report(event) -> str:
    pieces = [em.text for em in event.emissions if isinstance(em, TextEmission)]
    return "\n\n".join(pieces)


def event_file_actions(event) -> list:
    """Subset of emissions that are FileWrite / FileEdit."""
    return [
        em
        for em in event.emissions
        if isinstance(em, (FileWriteEmission, FileEditEmission))
    ]


# Aliases for callers asserting on an :class:`LLMResponse` rather than
# an :class:`ActionEvent`.  The shape of ``emissions`` is identical on
# both, so the same helpers apply.
response_code = event_code
response_thinking = event_thinking
response_title = event_title
response_terminal = event_terminal
response_report = event_report
response_file_actions = event_file_actions


def make_action_event(
    agent_name: str,
    *,
    thinking: str = "",
    code: str | None = None,
    title: str = "",
    report: str = "",
    file_actions: list | None = None,
    terminal: str | None = None,
    emissions: list | None = None,
    source: str = "main",
    **extra,
):
    """Build an :class:`ActionEvent` from legacy-shape kwargs.

    Mirror of :func:`make_response` but for the event-log side.  Tests
    that constructed ``ActionEvent(agent_name="a", thinking="...",
    code="...")`` stay readable via:

        make_action_event(agent_name="a", thinking="...", code="...")
    """
    from agex.agent.events import ActionEvent

    if emissions is not None:
        return ActionEvent(
            agent_name=agent_name,
            emissions=list(emissions),
            source=source,
            **extra,
        )

    built: list = []
    for fa in file_actions or []:
        built.append(_coerce_file_action(fa))

    title_or_none = title or None
    thinking_or_none = thinking or None

    if terminal is not None:
        built.append(
            TerminalEmission(
                commands=terminal, title=title_or_none, thinking=thinking_or_none
            )
        )
    elif code is not None:
        built.append(
            PythonEmission(code=code, title=title_or_none, thinking=thinking_or_none)
        )
    elif thinking:
        built.append(ThinkingEmission(text=thinking))

    if report:
        built.append(TextEmission(text=report))

    return ActionEvent(
        agent_name=agent_name,
        emissions=built,
        source=source,
        **extra,
    )
