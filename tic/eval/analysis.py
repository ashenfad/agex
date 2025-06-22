import ast


class FreeVariableAnalyzer(ast.NodeVisitor):
    """
    Finds free variables in a function's AST.

    A variable is "free" if it is read but not bound within the function's
    scope (as a parameter or a local assignment). This analyzer correctly
    handles nested functions and lambdas, propagating free variables up
    the scope chain.
    """

    def __init__(self, node: ast.FunctionDef | ast.Lambda):
        self.bound = set()
        self.loaded = set()
        self.globals = set()

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
        """Returns the set of free variables found."""
        return self.loaded - self.bound - self.globals

    def visit_Global(self, node: ast.Global):
        for name in node.names:
            self.globals.add(name)

    def visit_Nonlocal(self, node: ast.Nonlocal):
        # For our purpose, nonlocal behaves like global; it's not a free variable.
        for name in node.names:
            self.globals.add(name)

    def visit_Name(self, node: ast.Name):
        if node.id in self.globals:
            return

        if isinstance(node.ctx, ast.Load):
            if node.id not in self.bound:
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
