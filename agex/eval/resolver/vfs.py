from typing import TYPE_CHECKING, Any

from agex.eval.error import EvalError

from .base import BaseFinder, BaseLoader, ModuleSpec

if TYPE_CHECKING:
    from agex.agent.base import BaseAgent
    from agex.state.core import State


class VFSLoader(BaseLoader):
    """Loader for modules residing in the Virtual Filesystem."""

    def __init__(self, agent: "BaseAgent", session: str):
        self.agent = agent
        self.session = session

    def load(self, spec: ModuleSpec, state: "State") -> Any:
        from agex.eval.core import evaluate_program
        from agex.eval.objects import AgexModule, AgexVFSModule
        from agex.state import Namespaced

        # Get the underlying base store for namespacing
        base = state.base_store

        # Create isolated namespaced state: modules/<name>
        root_ns = Namespaced(base, "modules")
        module_state = Namespaced(root_ns, spec.name)

        # CLEAR OLD STATE: ensure clean slate for re-loading/overwriting
        # Preserve sub-namespaces (submodules) during reload!
        for key in list(module_state.keys()):
            # We check for AgexModule and AgexVFSModule to identify submodules
            # that were attached to this module's namespace.
            val = module_state.get(key)
            if not isinstance(val, (AgexModule, AgexVFSModule)):
                module_state.remove(key)

        # Execute code if it's not a pure namespace package
        code = ""
        is_package = spec.origin in ("vfs_package", "vfs_namespace")

        if spec.origin in ("vfs_file", "vfs_package") and spec.location:
            try:
                code_bytes = self.agent._fs_read(spec.location, self.session)
                code = code_bytes.decode("utf-8")
            except Exception as e:
                raise EvalError(
                    f"Failed to read module '{spec.name}' from VFS: {e}", None
                )

        # Compute package for relative imports:
        # - For packages (e.g., app/__init__.py): package is the module name itself
        # - For regular modules (e.g., app/views.py): package is the parent
        if is_package:
            package = spec.name
        elif "." in spec.name:
            package = spec.name.rsplit(".", 1)[0]
        else:
            package = ""

        # Set standard module attributes
        module_state.set("__name__", spec.name)
        module_state.set("__file__", spec.location or f"<virtual:{spec.name}>")
        module_state.set("__package__", package)
        if is_package:
            # Packages must have a __path__ attribute (list of strings)
            # For VFS packages, this is the directory containing __init__.py or the namespace dir
            path_val = spec.name.replace(".", "/")
            module_state.set("__path__", [path_val])

        try:
            evaluate_program(
                code,
                self.agent,
                state=module_state,
                session=self.session,
                package=package,
            )
        except Exception as e:
            from agex.agent.datatypes import _AgentExit

            if isinstance(e, _AgentExit):
                raise
            raise EvalError(
                f"Error initializing module '{spec.name}': {e}", None
            ) from e

        return AgexVFSModule(
            name=spec.name,
            state=module_state,
            agent_fingerprint=self.agent.fingerprint,
            session=self.session,
        )


class VFSFinder(BaseFinder):
    """Finder that checks the Virtual Filesystem for module code."""

    def __init__(self, agent: "BaseAgent", session: str):
        self.agent = agent
        self.session = session
        self.loader = VFSLoader(agent, session)

    def find_spec(self, fullname: str) -> ModuleSpec | None:
        # Map dotted name to directory/file path: pkg.sub -> pkg/sub
        path_prefix = fullname.replace(".", "/")

        # 1. Try Package (__init__.py)
        init_path = f"{path_prefix}/__init__.py"
        if self.agent._fs_exists(init_path, self.session):
            return ModuleSpec(
                name=fullname,
                origin="vfs_package",
                location=init_path,
                loader=self.loader,
            )

        # 2. Try Module (.py)
        py_path = f"{path_prefix}.py"
        if self.agent._fs_exists(py_path, self.session):
            return ModuleSpec(
                name=fullname, origin="vfs_file", location=py_path, loader=self.loader
            )

        # 3. Handle Namespace Package (dir exists but no code yet)
        if self.agent._fs_exists(f"{path_prefix}/", self.session):
            return ModuleSpec(
                name=fullname, origin="vfs_namespace", location=None, loader=self.loader
            )

        return None
