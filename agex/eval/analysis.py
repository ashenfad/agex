import ast


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

        # Parameters are always bound.
        args = node.args
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
