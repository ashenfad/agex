"""Tests for skill registration and discovery in the system message."""

import tempfile
from pathlib import Path
from tempfile import NamedTemporaryFile

from agex import Agent, clear_agent_registry
from agex.llm.dummy_client import Dummy
from tests.agex._emissions import make_response

_counter = 0


def setup_module():
    clear_agent_registry()


def teardown_module():
    clear_agent_registry()


def _make_agent():
    global _counter
    _counter += 1
    return Agent(name=f"skill_agent_{_counter}")


def test_no_skills_no_section():
    """System message omits skills section when no skills exist."""
    agent = _make_agent()
    msg = agent._build_system_message()
    assert "# Skills" not in msg


def test_skill_from_bytes():
    """Skills registered as raw bytes show name and description."""
    agent = _make_agent()
    agent.skill(b"---\nname: my-lib\ndescription: A useful library\n---\n\n# my-lib\n")

    msg = agent._build_system_message()
    assert "# Skills" in msg
    assert "my-lib: A useful library" in msg
    assert "cat /skills/<name>/SKILL.md" in msg


def test_skill_from_path():
    """Skills registered from a Path-like object work."""
    agent = _make_agent()
    with NamedTemporaryFile(suffix=".md", delete=False) as f:
        f.write(b"---\nname: from-file\ndescription: File skill\n---\n")
        f.flush()
        agent.skill(Path(f.name))

    msg = agent._build_system_message()
    assert "from-file: File skill" in msg


def test_skill_without_frontmatter():
    """Skills without frontmatter fall back to fallback name."""
    agent = _make_agent()
    agent.skill(b"# Just some docs\n")

    msg = agent._build_system_message()
    assert "# Skills" in msg
    assert "- skill" in msg


def test_multiple_skills_sorted():
    """Multiple skills are listed in sorted order."""
    agent = _make_agent()
    agent.skill(b"---\nname: zeta\ndescription: Last one\n---\n")
    agent.skill(b"---\nname: alpha\ndescription: First one\n---\n")

    msg = agent._build_system_message()
    alpha_pos = msg.index("alpha: First one")
    zeta_pos = msg.index("zeta: Last one")
    assert alpha_pos < zeta_pos


def test_skill_name_from_frontmatter_overrides_fallback():
    """Frontmatter name overrides the fallback name."""
    agent = _make_agent()
    agent.skill(b"---\nname: display-name\ndescription: Overridden\n---\n")

    msg = agent._build_system_message()
    assert "display-name: Overridden" in msg


def test_skill_parent_dir_name():
    """SKILL.md files use parent directory name as fallback."""
    agent = _make_agent()
    # Simulate importlib.resources Traversable with parent.name
    with NamedTemporaryFile(prefix="SKILL", suffix=".md", dir=None, delete=False) as f:
        f.write(b"# No frontmatter\n")
        f.flush()
        path = Path(f.name)

    # Rename to SKILL.md in a named directory
    skill_dir = path.parent / "my-cool-lib"
    skill_dir.mkdir(exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    path.rename(skill_path)

    try:
        agent.skill(skill_path)
        msg = agent._build_system_message()
        assert "- my-cool-lib" in msg
    finally:
        skill_path.unlink(missing_ok=True)
        skill_dir.rmdir()


def test_skill_name_slugified():
    """Skill names with spaces/special chars are coerced to path-safe slugs."""
    agent = _make_agent()
    agent.skill(b"---\nname: My Cool Library!\ndescription: Has spaces\n---\n")

    msg = agent._build_system_message()
    assert "my-cool-library" in msg
    # Verify no unsafe characters remain
    assert "My Cool Library!" not in msg


def test_skill_readable_in_task():
    """Agent can read a skill file via open() during task execution."""
    llm = Dummy(
        responses=[
            make_response(
                thinking="Read the skill file.",
                code=(
                    'with open("/skills/test-lib/SKILL.md") as f:\n'
                    "    content = f.read()\n"
                    "task_success(content)"
                ),
            )
        ]
    )
    agent = Agent(name="skill_e2e", llm=llm)
    agent.skill(
        b"---\nname: test-lib\ndescription: A test skill\n---\n\n# test-lib\n\nUse it wisely.\n"
    )

    @agent.task
    def read_skill() -> str:  # type: ignore[return-value]
        """Read a skill file and return its content."""
        pass

    result = read_skill()
    assert "# test-lib" in result
    assert "Use it wisely." in result


def test_skill_listdir_visible_in_task():
    """The /skills directory is visible via os.listdir during task execution."""
    llm = Dummy(
        responses=[
            make_response(
                thinking="List the skills directory.",
                code=(
                    "import os\nentries = os.listdir('/skills')\ntask_success(entries)"
                ),
            )
        ]
    )
    agent = Agent(name="skill_ls_e2e", llm=llm)
    agent.skill(b"---\nname: alpha-lib\n---\n# alpha\n")
    agent.skill(b"---\nname: beta-lib\n---\n# beta\n")

    @agent.task
    def list_skills() -> list:  # type: ignore[return-value]
        """List available skill directories."""
        pass

    result = list_skills()
    assert "alpha-lib" in result
    assert "beta-lib" in result


def test_skill_multiline_description():
    """Multi-line YAML descriptions are joined into a single line."""
    agent = _make_agent()
    agent.skill(b"---\nname: multi\ndescription: |\n  Line one\n  line two\n---\n")

    msg = agent._build_system_message()
    assert "multi: Line one line two" in msg


def test_skill_multiline_description_inline():
    """Multi-line description without block scalar indicator."""
    agent = _make_agent()
    agent.skill(b"---\nname: multi2\ndescription: First line\n  continued here\n---\n")

    msg = agent._build_system_message()
    assert "multi2: First line continued here" in msg


def test_skill_rejects_string():
    """skill() rejects plain strings."""
    agent = _make_agent()
    try:
        agent.skill("not a path")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


# --- Directory-based skill tests ---


def _make_skill_dir(tmp_path: Path, name: str = "my-dsl") -> Path:
    """Helper to create a skill directory with SKILL.md and sibling files."""
    skill_dir = tmp_path / name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_bytes(
        b"---\nname: my-dsl\ndescription: A DSL reference\n---\n\n# My DSL\n\nStart here.\n"
    )
    (skill_dir / "types.md").write_bytes(b"# Types\n\nType reference docs.\n")
    (skill_dir / "examples.md").write_bytes(b"# Examples\n\nSome examples.\n")
    return skill_dir


def test_skill_from_directory():
    """Registering a directory mounts all files and shows skill in system message."""
    agent = _make_agent()
    with tempfile.TemporaryDirectory() as tmp:
        skill_dir = _make_skill_dir(Path(tmp))
        agent.skill(skill_dir)

    msg = agent._build_system_message()
    assert "my-dsl: A DSL reference" in msg


def test_skill_directory_missing_skill_md():
    """Directory without SKILL.md raises ValueError."""
    agent = _make_agent()
    with tempfile.TemporaryDirectory() as tmp:
        skill_dir = Path(tmp) / "bad-skill"
        skill_dir.mkdir()
        (skill_dir / "other.md").write_bytes(b"# Not a SKILL.md\n")
        try:
            agent.skill(skill_dir)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "SKILL.md" in str(e)


def test_skill_directory_nested_files():
    """Nested subdirectories are preserved in the VFS."""
    agent = _make_agent()
    with tempfile.TemporaryDirectory() as tmp:
        skill_dir = Path(tmp) / "nested-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_bytes(
            b"---\nname: nested\ndescription: Has subdirs\n---\n"
        )
        sub = skill_dir / "reference"
        sub.mkdir()
        (sub / "operators.md").write_bytes(b"# Operators\n")
        agent.skill(skill_dir)

    # Verify via VFS creation
    from agex.fs.skills_vfs import create_skills_fs

    vfs = create_skills_fs(agent._skills)
    assert vfs is not None
    # The nested file should be accessible
    assert vfs.read("nested/reference/operators.md") == b"# Operators\n"
    assert vfs.read("nested/SKILL.md").startswith(b"---")


def test_skill_directory_skips_dotfiles():
    """Dotfiles and dotdirectories are excluded from directory skills."""
    agent = _make_agent()
    with tempfile.TemporaryDirectory() as tmp:
        skill_dir = Path(tmp) / "dottest"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_bytes(b"---\nname: dottest\n---\n")
        (skill_dir / ".hidden").write_bytes(b"secret")
        dot_dir = skill_dir / ".git"
        dot_dir.mkdir()
        (dot_dir / "config").write_bytes(b"gitconfig")
        agent.skill(skill_dir)

    from agex.fs.skills_vfs import create_skills_fs

    vfs = create_skills_fs(agent._skills)
    assert vfs is not None
    assert vfs.read("dottest/SKILL.md").startswith(b"---")
    try:
        vfs.read("dottest/.hidden")
        assert False, "Should not find dotfile"
    except (FileNotFoundError, OSError):
        pass
    try:
        vfs.read("dottest/.git/config")
        assert False, "Should not find dotdir contents"
    except (FileNotFoundError, OSError):
        pass


def test_skill_directory_name_from_frontmatter():
    """Frontmatter name takes precedence over directory name."""
    agent = _make_agent()
    with tempfile.TemporaryDirectory() as tmp:
        skill_dir = Path(tmp) / "dir-name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_bytes(
            b"---\nname: frontmatter-name\ndescription: FM wins\n---\n"
        )
        agent.skill(skill_dir)

    msg = agent._build_system_message()
    assert "frontmatter-name: FM wins" in msg
    assert "dir-name" not in msg


def test_skill_directory_name_fallback_to_dirname():
    """Without frontmatter name, directory name is used."""
    agent = _make_agent()
    with tempfile.TemporaryDirectory() as tmp:
        skill_dir = Path(tmp) / "my-fallback-dir"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_bytes(b"# No frontmatter\n")
        agent.skill(skill_dir)

    msg = agent._build_system_message()
    assert "- my-fallback-dir" in msg


def test_skill_directory_siblings_readable_in_task():
    """Agent can read sibling files from a directory skill during task execution."""
    llm = Dummy(
        responses=[
            make_response(
                thinking="Read the sibling file.",
                code=(
                    'with open("/skills/my-dsl/types.md") as f:\n'
                    "    content = f.read()\n"
                    "task_success(content)"
                ),
            )
        ]
    )
    agent = Agent(name="skill_dir_e2e", llm=llm)

    with tempfile.TemporaryDirectory() as tmp:
        skill_dir = _make_skill_dir(Path(tmp))
        agent.skill(skill_dir)

    @agent.task
    def read_sibling() -> str:  # type: ignore[return-value]
        """Read a sibling file from the skill directory."""
        pass

    result = read_sibling()
    assert "Type reference docs." in result
