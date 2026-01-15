import ast
from typing import Any


class FreeVariableAnalyzer(ast.NodeVisitor):
    """
    Finds free variables in a function's AST.

    A variable is "free" if it is read but not bound within the function's
    scope (as a parameter or a local assignment). This analyzer correctly
    handles nested functions and lambdas, propagating free variables up
    the scope chain.

    Improved to exclude builtins and handle Python's scoping rules properly.
    """

    def __init__(self, node: ast.FunctionDef | ast.Lambda):
        self.bound = set()
        self.loaded = set()
        self.globals = set()
        self.exception_vars = set()  # Variables bound in except clauses
        self.default_refs = (
            set()
        )  # Variables referenced in default parameter values (always free)

        # Visit default parameter values FIRST (they reference outer scope, before params are bound)
        args = node.args
        for i, default in enumerate(args.defaults):
            # Track references in defaults separately - these are always to outer scope
            old_loaded = self.loaded.copy()
            self.visit(default)
            self.default_refs.update(self.loaded - old_loaded)
        for default in args.kw_defaults:
            if default is not None:  # kw_defaults can contain None
                # Track references in defaults separately - these are always to outer scope
                old_loaded = self.loaded.copy()
                self.visit(default)
                self.default_refs.update(self.loaded - old_loaded)

        # Parameters are bound AFTER visiting defaults
        for arg in args.args:
            self.bound.add(arg.arg)
        for arg in args.kwonlyargs:
            self.bound.add(arg.arg)
        if args.vararg:
            self.bound.add(args.vararg.arg)
        if args.kwarg:
            self.bound.add(args.kwarg.arg)

        # Visit the function body to find all other bindings and loads.
        if isinstance(node.body, list):  # FunctionDef
            for stmt in node.body:
                self.visit(stmt)
        else:  # Lambda
            self.visit(node.body)

    @property
    def free(self) -> set[str]:
        """Returns the set of free variables found, excluding builtins."""
        # Get the basic free variables (loaded but not bound/global)
        basic_free = self.loaded - self.bound - self.globals - self.exception_vars

        # Add variables from default parameters - these are always free variables
        # even if they match parameter names (they refer to outer scope)
        basic_free = basic_free | self.default_refs

        # Exclude builtins - these should resolve through the builtin system, not be captured
        from ..eval.builtins import BUILTINS, STATEFUL_BUILTINS

        builtins_set = set(BUILTINS.keys()) | set(STATEFUL_BUILTINS.keys())

        # Return only variables that are truly free (not builtins)
        return basic_free - builtins_set

    def visit_Global(self, node: ast.Global):
        for name in node.names:
            self.globals.add(name)

    def visit_Nonlocal(self, node: ast.Nonlocal):
        # For our purpose, nonlocal behaves like global; it's not a free variable.
        for name in node.names:
            self.globals.add(name)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        # Handle exception variables properly - they're bound in the except block
        if node.name:
            self.exception_vars.add(node.name)

        # Continue visiting the except block body
        for stmt in node.body:
            self.visit(stmt)

    def visit_Name(self, node: ast.Name):
        if node.id in self.globals:
            return

        if isinstance(node.ctx, ast.Load):
            if node.id not in self.bound and node.id not in self.exception_vars:
                self.loaded.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.bound.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # First, bind the function's own name in the current scope.
        self.bound.add(node.name)
        # Then, analyze the nested function to see what free variables it has.
        # Any variable that is free in the nested function is considered "loaded"
        # by the outer function.
        analyzer = FreeVariableAnalyzer(node)
        for free_var in analyzer.free:
            if free_var not in self.bound:
                self.loaded.add(free_var)

    def visit_Lambda(self, node: ast.Lambda):
        # Lambdas are analyzed for free variables just like nested functions.
        analyzer = FreeVariableAnalyzer(node)
        for free_var in analyzer.free:
            if free_var not in self.bound:
                self.loaded.add(free_var)


def get_free_variables(node: ast.FunctionDef | ast.Lambda) -> set[str]:
    """A helper function to analyze a function or lambda node for free variables."""
    return FreeVariableAnalyzer(node).free


def render_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Render a function signature as a Python-like string from an AST node."""
    args = []

    # Positional args
    for arg in node.args.args:
        arg_str = arg.arg
        if arg.annotation:
            arg_str += f": {ast.unparse(arg.annotation)}"
        args.append(arg_str)

    # *args
    if node.args.vararg:
        arg_str = f"*{node.args.vararg.arg}"
        if node.args.vararg.annotation:
            arg_str += f": {ast.unparse(node.args.vararg.annotation)}"
        args.append(arg_str)

    # Keyword-only args
    for arg in node.args.kwonlyargs:
        arg_str = arg.arg
        if arg.annotation:
            arg_str += f": {ast.unparse(arg.annotation)}"
        args.append(arg_str)

    # **kwargs
    if node.args.kwarg:
        arg_str = f"**{node.args.kwarg.arg}"
        if node.args.kwarg.annotation:
            arg_str += f": {ast.unparse(node.args.kwarg.annotation)}"
        args.append(arg_str)

    ret_ann = ""
    if node.returns:
        ret_ann = f" -> {ast.unparse(node.returns)}"

    return f"def {node.name}({', '.join(args)}){ret_ann}:"


def get_workspace_recap(agent: Any, session: str = "default") -> str:
    """Scan the VFS for .py files and generate a summary of their contents."""
    if not agent._fs_config:
        return ""

    vfs = agent.fs(session)
    # We only want .py files
    py_files = [f for f in vfs.list("/", recursive=True) if f.endswith(".py")]
    if not py_files:
        return ""

    recap = ["[Current Workspace Inventory]"]

    for file_path in sorted(py_files):
        try:
            content = vfs.read(file_path).decode("utf-8")
            tree = ast.parse(content)

            recap.append(f"- {file_path}:")
            found_anything = False

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sig = render_signature(node)
                    recap.append(f"    {sig}")
                    doc = ast.get_docstring(node)
                    if doc:
                        # Keep docstrings short
                        first_line = doc.split("\n")[0]
                        recap.append(f'        """{first_line}"""')
                    found_anything = True
                elif isinstance(node, ast.ClassDef):
                    recap.append(f"    class {node.name}:")
                    doc = ast.get_docstring(node)
                    if doc:
                        first_line = doc.split("\n")[0]
                        recap.append(f'        """{first_line}"""')
                    # Optionally list methods? Keep it brief for now.
                    found_anything = True

            if not found_anything:
                recap.append("    (empty or constants only)")

        except Exception:
            # Skip files that fail to parse
            continue

    return "\n".join(recap)
