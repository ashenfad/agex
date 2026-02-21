import pytest

from agex import Agent, connect_fs, connect_state
from agex.agent.base import clear_agent_registry
from agex.eval.core import run_file_in_sandbox
from agex.llm import Dummy, LLMResponse


@pytest.fixture(autouse=True)
def cleanup():
    clear_agent_registry()
    yield
    clear_agent_registry()


def create_agent():
    return Agent(
        llm=Dummy(),
        fs=connect_fs(type="virtual"),
        state=connect_state(type="versioned", storage="memory"),
    )


class TestRelativeImportsInRunFile:
    """Test relative imports when using run_file_in_sandbox."""

    def test_from_dot_import_module(self):
        """Test `from . import views` in app/main.py."""
        agent = create_agent()
        fs = agent.fs()

        fs.write("app/views.py", b"VAL = 42")
        fs.write(
            "app/main.py",
            b"""
from . import views
result = views.VAL
""",
        )

        state = run_file_in_sandbox(agent, "app/main.py")
        assert state.get("result") == 42

    def test_from_dot_module_import_name(self):
        """Test `from .views import VAL` in app/main.py."""
        agent = create_agent()
        fs = agent.fs()

        fs.write("app/views.py", b"VAL = 100")
        fs.write(
            "app/main.py",
            b"""
from .views import VAL
result = VAL
""",
        )

        state = run_file_in_sandbox(agent, "app/main.py")
        assert state.get("result") == 100

    def test_from_dot_import_multiple(self):
        """Test `from . import views, utils` in app/main.py."""
        agent = create_agent()
        fs = agent.fs()

        fs.write("app/views.py", b"X = 1")
        fs.write("app/utils.py", b"Y = 2")
        fs.write(
            "app/main.py",
            b"""
from . import views, utils
result = views.X + utils.Y
""",
        )

        state = run_file_in_sandbox(agent, "app/main.py")
        assert state.get("result") == 3

    def test_from_dotdot_import(self):
        """Test `from .. import shared` in app/sub/module.py."""
        agent = create_agent()
        fs = agent.fs()

        fs.write("app/shared.py", b"SHARED = 'hello'")
        fs.write(
            "app/sub/module.py",
            b"""
from .. import shared
result = shared.SHARED
""",
        )

        state = run_file_in_sandbox(agent, "app/sub/module.py")
        assert state.get("result") == "hello"

    def test_from_dotdot_module_import(self):
        """Test `from ..utils import helper` in app/sub/module.py."""
        agent = create_agent()
        fs = agent.fs()

        fs.write("app/utils.py", b"def helper(): return 'helped'")
        fs.write(
            "app/sub/module.py",
            b"""
from ..utils import helper
result = helper()
""",
        )

        state = run_file_in_sandbox(agent, "app/sub/module.py")
        assert state.get("result") == "helped"

    def test_relative_import_with_alias(self):
        """Test `from . import views as v`."""
        agent = create_agent()
        fs = agent.fs()

        fs.write("app/views.py", b"VAL = 99")
        fs.write(
            "app/main.py",
            b"""
from . import views as v
result = v.VAL
""",
        )

        state = run_file_in_sandbox(agent, "app/main.py")
        assert state.get("result") == 99

    def test_relative_import_in_top_level_fails(self):
        """Test that relative import in top-level file fails gracefully."""
        agent = create_agent()
        fs = agent.fs()

        fs.write("main.py", b"from . import something")

        with pytest.raises(Exception) as exc:
            run_file_in_sandbox(agent, "main.py")

        assert (
            "relative import" in str(exc.value).lower()
            or "non-package" in str(exc.value).lower()
        )

    def test_relative_import_beyond_top_level_fails(self):
        """Test that going too far up fails."""
        agent = create_agent()
        fs = agent.fs()

        fs.write("app/main.py", b"from .. import something")

        with pytest.raises(Exception) as exc:
            run_file_in_sandbox(agent, "app/main.py")

        error_msg = str(exc.value).lower()
        assert (
            "beyond" in error_msg
            or "top-level" in error_msg
            or "no parent package" in error_msg
            or "resolved from relative import" in error_msg
        )


class TestRelativeImportsInVFSModules:
    """Test relative imports within VFS modules loaded via import."""

    def test_vfs_module_relative_import(self):
        """Test that a VFS module can use relative imports."""
        agent = create_agent()
        fs = agent.fs()

        fs.write("pkg/utils.py", b"UTIL_VAL = 10")
        fs.write(
            "pkg/main.py",
            b"""
from . import utils
def get_val():
    return utils.UTIL_VAL
""",
        )

        agent.llm.responses = [
            LLMResponse(
                thinking="test",
                code="import pkg.main\ntask_success(pkg.main.get_val())",
            )
        ]

        @agent.task
        def task():
            """Test task."""
            pass

        assert task() == 10

    def test_nested_vfs_module_relative_import(self):
        """Test relative imports in nested VFS modules."""
        agent = create_agent()
        fs = agent.fs()

        fs.write("pkg/shared.py", b"SHARED = 'shared_value'")
        fs.write(
            "pkg/sub/module.py",
            b"""
from .. import shared
def get_shared():
    return shared.SHARED
""",
        )

        agent.llm.responses = [
            LLMResponse(
                thinking="test",
                code="import pkg.sub.module\ntask_success(pkg.sub.module.get_shared())",
            )
        ]

        @agent.task
        def task():
            """Test task."""
            pass

        assert task() == "shared_value"


class TestResolveRelativeImport:
    """Unit tests for the resolve_relative_import helper."""

    def test_level_1_with_module(self):
        from agex.eval.statements import resolve_relative_import

        assert resolve_relative_import("app", 1, "views") == "app.views"
        assert resolve_relative_import("app.sub", 1, "utils") == "app.sub.utils"

    def test_level_1_without_module(self):
        from agex.eval.statements import resolve_relative_import

        assert resolve_relative_import("app", 1, None) == "app"
        assert resolve_relative_import("app.sub", 1, None) == "app.sub"

    def test_level_2_with_module(self):
        from agex.eval.statements import resolve_relative_import

        assert resolve_relative_import("app.sub", 2, "utils") == "app.utils"
        assert resolve_relative_import("app.sub.deep", 2, "other") == "app.sub.other"

    def test_level_2_without_module(self):
        from agex.eval.statements import resolve_relative_import

        assert resolve_relative_import("app.sub", 2, None) == "app"
        assert resolve_relative_import("app.sub.deep", 2, None) == "app.sub"

    def test_no_package_fails(self):
        from agex.eval.statements import resolve_relative_import

        with pytest.raises(ValueError, match="non-package"):
            resolve_relative_import("", 1, "views")

    def test_beyond_top_level_fails(self):
        from agex.eval.statements import resolve_relative_import

        with pytest.raises(ValueError, match="beyond|parent"):
            resolve_relative_import("app", 2, "something")
        with pytest.raises(ValueError, match="beyond|parent"):
            resolve_relative_import("app", 3, None)
