"""Tests for skill discovery in the system message."""

from agex import Agent, clear_agent_registry, connect_fs, connect_state

_counter = 0


def setup_module():
    clear_agent_registry()


def teardown_module():
    clear_agent_registry()


def _make_agent():
    global _counter
    _counter += 1
    return Agent(
        name=f"skill_agent_{_counter}",
        fs=connect_fs(type="virtual"),
        state=connect_state(type="versioned", storage="memory"),
    )


def _write_skill(agent, path, content):
    """Write a file to the agent's VFS and commit so discovery can see it."""
    fs = agent.fs()
    fs.write(path, content)
    agent.state().commit()


def test_no_skills_no_section():
    """System message omits skills section when no skills exist."""
    agent = _make_agent()
    msg = agent._build_system_message()
    assert "# Skills" not in msg


def test_skill_with_frontmatter():
    """Skills with YAML frontmatter show name and description."""
    agent = _make_agent()
    _write_skill(
        agent,
        "skills/my-lib/SKILL.md",
        b"---\nname: my-lib\ndescription: A useful library\n---\n\n# my-lib\n",
    )

    msg = agent._build_system_message()
    assert "# Skills" in msg
    assert "my-lib: A useful library" in msg
    assert "cat /skills/<name>/SKILL.md" in msg


def test_skill_without_frontmatter():
    """Skills without frontmatter fall back to directory name."""
    agent = _make_agent()
    _write_skill(agent, "skills/raw-skill/SKILL.md", b"# Just some docs\n")

    msg = agent._build_system_message()
    assert "# Skills" in msg
    assert "- raw-skill" in msg


def test_multiple_skills_sorted():
    """Multiple skills are listed in sorted order."""
    agent = _make_agent()
    _write_skill(
        agent,
        "skills/zeta/SKILL.md",
        b"---\nname: zeta\ndescription: Last one\n---\n",
    )
    _write_skill(
        agent,
        "skills/alpha/SKILL.md",
        b"---\nname: alpha\ndescription: First one\n---\n",
    )

    msg = agent._build_system_message()
    alpha_pos = msg.index("alpha: First one")
    zeta_pos = msg.index("zeta: Last one")
    assert alpha_pos < zeta_pos


def test_skill_name_override():
    """Frontmatter name overrides directory name in listing."""
    agent = _make_agent()
    _write_skill(
        agent,
        "skills/dir-name/SKILL.md",
        b"---\nname: display-name\ndescription: Overridden\n---\n",
    )

    msg = agent._build_system_message()
    assert "display-name: Overridden" in msg


def test_no_fs_configured():
    """Agent without filesystem skips skills gracefully."""
    global _counter
    _counter += 1
    agent = Agent(name=f"no_fs_agent_{_counter}", fs=None)
    msg = agent._build_system_message()
    assert "# Skills" not in msg


def test_non_skill_files_ignored():
    """Files outside the SKILL.md convention are ignored."""
    agent = _make_agent()
    _write_skill(agent, "skills/my-lib/README.md", b"# Not a skill\n")

    msg = agent._build_system_message()
    assert "# Skills" not in msg
