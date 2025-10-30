from datetime import datetime, timezone
from pathlib import Path

from agex.agent import BaseAgent
from agex.agent.fingerprint import compute_agent_fingerprint_from_policy
from agex.llm.core import LLMClient
from agex.render.definitions import render_definitions


def _safe_slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in text)[:50]


def _cache_path(
    agent_name: str, fingerprint: str, target_chars: int, model_id: str
) -> Path:
    cache_dir = Path(".agex/primer_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = _safe_slug(agent_name or "agent")
    model_slug = _safe_slug(model_id or "default")
    return cache_dir / f"{name}-{fingerprint[:8]}-ch{target_chars}-m{model_slug}.md"


def summarize_capabilities(
    agent: BaseAgent,
    target_chars: int,
    llm_client: LLMClient | None = None,
    use_cache: bool = True,
) -> str:
    """
    Build a concise capabilities primer from the agent's registered capabilities.

    - Honors visibility via render_definitions(agent)
    - Uses llm_client.summarize to compress into ~token_budget tokens (best-effort)
    - Optional on-disk cache under .agex/primer_cache/
    """
    # Resolve summarizer client and model id for cache key
    client = llm_client or agent.llm_client
    try:
        model_id = client.model
    except Exception:
        model_id = "default"

    # Compute fingerprint for current registrations
    fingerprint = compute_agent_fingerprint_from_policy(agent)

    # Check cache
    cache_file = _cache_path(
        getattr(agent, "name", "agent"), fingerprint, target_chars, model_id
    )
    if use_cache and cache_file.exists():
        try:
            return cache_file.read_text(encoding="utf-8")
        except Exception:
            pass

    # Render current registrations (visibility-aware)
    rendered = render_definitions(agent)

    # Build prompt content
    system_instructions = (
        "You are writing a thorough capabilities primer (markdown format) for an agent.\n"
        "The agent has access to a restricted set of Python capabilities.\n"
        "Summarize the patterns of use for the capabilities below,\n"
        "Deduplicate similar items or repeated documentation.\n"
        "Give actionable guidance and 1-2 canonical usage snippets per cluster.\n"
        "Please only mention modules defined within these capabilities.\n"
        "Do not invent functions or classes that are not present.\n"
        "Do not suggest decorators, nonlocals, globals, async, or await (not supported in this environment).\n"
        f"Write AT LEAST {target_chars} characters. If needed, expand each section with more detail and examples."
    )

    # Provide definitions in a fenced block to reduce bleeding
    user_content = "Registered API (visibility-filtered):\n\n```\n" + rendered + "\n```"

    # Generate summary text
    summary_text = client.summarize(system=system_instructions, content=user_content)

    # Prepend a small header for provenance
    created = datetime.now(timezone.utc).isoformat()
    header = (
        f"# Capabilities Primer\n"
        f"# agent: {getattr(agent, 'name', 'agent')}\n"
        f"# fingerprint: {fingerprint[:8]}\n"
        f"# target_chars: {target_chars}\n"
        f"# model: {model_id}\n"
        f"# created_at: {created}\n\n"
    )
    result = header + summary_text.strip() + "\n"

    # Write cache
    if use_cache:
        try:
            cache_file.write_text(result, encoding="utf-8")
        except Exception:
            pass

    return result
