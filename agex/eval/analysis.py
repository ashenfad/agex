import ast
from typing import Any


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

    recap = []

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

                    # List methods
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            # Skip private methods
                            if item.name.startswith("_") and not item.name.startswith(
                                "__"
                            ):
                                continue
                            method_sig = render_signature(item)
                            recap.append(f"        {method_sig}")

                    found_anything = True

            if not found_anything:
                recap.append("    (empty or constants only)")

        except Exception:
            # Skip files that fail to parse
            continue

    return "\n".join(recap)
