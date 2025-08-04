"""
Message formatting utilities for agent communication.

This module handles the conversion of plain Python content (from renderers)
into markdown-formatted messages suitable for LLM consumption.
"""


def format_context_as_markdown(context: str) -> str:
    """
    Format plain Python context content as markdown for LLM consumption.

    Takes the output from ContextRenderer (plain Python) and wraps it in
    appropriate markdown formatting with code blocks and headers.

    Args:
        context: Plain text context from ContextRenderer

    Returns:
        Markdown-formatted string suitable for LLM system messages
    """
    if not context.strip():
        return ""

    lines = context.split("\n")
    sections = []
    current_section = []
    current_header = None

    for line in lines:
        # Detect section headers (like "Agent stdout:")
        if line.endswith(":") and not line.startswith(" ") and "=" not in line:
            # Save previous section if exists
            if current_section:
                if current_header:
                    sections.append(_format_section(current_header, current_section))
                else:
                    # No previous header, treat as state changes
                    sections.append(_format_section("State Changes", current_section))

            # Start new section
            current_header = line[:-1]  # Remove trailing colon
            current_section = []
        else:
            current_section.append(line)

    # Add final section
    if current_section:
        if current_header:
            sections.append(_format_section(current_header, current_section))
        else:
            # No header found, treat as state changes
            sections.append(_format_section("State Changes", current_section))

    return "\n\n".join(sections)


def _format_section(header: str, lines: list[str]) -> str:
    """Format a section with header and Python code block."""
    # Clean up header formatting
    if header == "Agent stdout":
        header = "## Stdout (prints and errors)..."
    elif "=" in "".join(lines):  # Likely state changes
        header = "## State Changes"
    else:
        header = f"## {header}"

    # Clean up content - remove quotes from printed strings
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove quotes around printed strings (but keep them for state assignments)
        if "=" not in line and line.startswith("'") and line.endswith("'"):
            line = line[1:-1]  # Remove surrounding quotes
        elif "=" not in line and line.startswith('"') and line.endswith('"'):
            line = line[1:-1]  # Remove surrounding quotes

        cleaned_lines.append(line)

    if not cleaned_lines:
        return ""

    # Format as markdown with Python code block
    content = "\n".join(cleaned_lines)
    return f"{header}\n\n```python\n{content}\n```"
