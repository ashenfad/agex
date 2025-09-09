from __future__ import annotations

from typing import Any, Dict

from agex.agent.primer_text import BUILTIN_PRIMER
from agex.render.definitions import render_definitions
from agex.tokenizers.tiktoken import TiktokenTokenizer


def _count_tokens(text: str, model_name: str) -> int:
    tokenizer = TiktokenTokenizer(model_name=model_name)
    return len(tokenizer.encode(text))


def system_token_count(agent, model_name: str = "gpt-4") -> Dict[str, Any]:
    """
    Estimate token usage for the agent's static system context before a call.

    Sections counted:
    - builtin_primer: The framework's built-in primer text
    - registered_resources: Header + rendered definitions based on visibility
    - agent_primer: The agent-specific primer string (if any)

    Returns a breakdown by section and a total.
    """
    # 1) Builtin primer
    builtin_primer_text = BUILTIN_PRIMER or ""
    builtin_tokens = (
        _count_tokens(builtin_primer_text, model_name) if builtin_primer_text else 0
    )

    # 2) Registered resources (header + rendered definitions)
    rendered_defs = render_definitions(agent)  # respects visibility
    if rendered_defs.strip():
        resources_text = "# Registered Resources\n\n" + rendered_defs
        resources_tokens = _count_tokens(resources_text, model_name)
    else:
        resources_tokens = 0

    # 3) Agent primer
    agent_primer_text = getattr(agent, "primer", None) or ""
    agent_primer_tokens = (
        _count_tokens(agent_primer_text, model_name) if agent_primer_text else 0
    )

    by_section = {
        "builtin_primer": builtin_tokens,
        "registered_resources": resources_tokens,
        "agent_primer": agent_primer_tokens,
    }
    total = sum(by_section.values())

    return {
        "model": model_name,
        "by_section": by_section,
        "total": total,
    }
