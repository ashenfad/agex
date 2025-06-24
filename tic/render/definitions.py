import dataclasses
import inspect

from ..agent import (
    Agent,
    RegisteredClass,
    RegisteredFn,
    RegisteredModule,
    Visibility,
)


def render_definitions(agent: Agent, full: bool = False) -> str:
    """
    Renders the registered functions, classes, and modules of an agent
    into a Python-like string of signatures and docstrings.
    If `full` is True, all members are rendered regardless of visibility.
    """
    output = []
    # Render standalone functions
    for name, spec in agent.fn_registry.items():
        if not full and spec.visibility != "high":
            continue
        output.append(_render_function(name, spec, full=full))

    # Render classes
    for name, spec in agent.cls_registry.items():
        # The top-level rendering logic for a class is now more complex,
        # because a low-visibility class might be "promoted" if it has a
        # high-visibility member.
        is_promoted = _is_class_promoted(spec)
        effective_visibility = spec.visibility
        if spec.visibility == "low" and is_promoted:
            effective_visibility = "medium"

        if full or effective_visibility != "low":
            # We create a new spec with the effective visibility to pass down.
            spec_to_render = dataclasses.replace(spec, visibility=effective_visibility)
            output.append(_render_class(name, spec_to_render, full=full))

    # Render modules
    for name, spec in agent.importable_modules.items():
        rendered_module = _render_module(name, spec, full=full)
        if rendered_module:
            output.append(rendered_module)

    return "\n\n".join(output)


def _should_render_member(
    member_vis: Visibility, container_vis: Visibility, full: bool = False
) -> bool:
    """Determines if a member should be rendered based on its and its container's visibility."""
    if full:
        return True
    if member_vis == "high":
        return True
    if member_vis == "medium" and container_vis == "high":
        return True
    return False


def _is_class_promoted(spec: RegisteredClass) -> bool:
    """A class is promoted if it contains any high-visibility members."""
    return any(m.visibility == "high" for m in spec.methods.values()) or any(
        a.visibility == "high" for a in spec.attrs.values()
    )


def _is_module_promoted(spec: RegisteredModule) -> bool:
    """A module is promoted if it contains any high-visibility functions or (potentially promoted) classes."""
    if any(f.visibility == "high" for f in spec.fns.values()):
        return True
    for cls_spec in spec.classes.values():
        if cls_spec.visibility == "high" or _is_class_promoted(cls_spec):
            return True
    return False


def _render_module(name: str, spec: RegisteredModule, full: bool = False) -> str:
    """Renders a single module definition based on its visibility and its members' visibilities."""
    # Determine the module's effective visibility. A low-vis module can be
    # "promoted" to medium-vis if it contains a high-vis member.
    is_promoted = _is_module_promoted(spec)
    effective_visibility = spec.visibility
    if spec.visibility == "low" and is_promoted:
        effective_visibility = "medium"

    # Don't render low-visibility modules at all, unless in full mode.
    if not full and effective_visibility == "low":
        return ""

    output = [f"module {name}:"]
    indent = "    "
    rendered_fns = []
    rendered_classes = []

    # Render functions
    for fn_name, fn_member_spec in spec.fns.items():
        if _should_render_member(
            fn_member_spec.visibility or spec.visibility,
            effective_visibility,
            full=full,
        ):
            fn = getattr(spec.module, fn_name)
            doc = (
                fn_member_spec.docstring
                if fn_member_spec.docstring is not None
                else fn.__doc__
            )
            # We pass the member's own visibility down so the function renderer
            # knows whether to render the docstring or not.
            fn_spec = RegisteredFn(
                fn=fn,
                docstring=doc,
                visibility=fn_member_spec.visibility or spec.visibility,
            )
            rendered_fns.append(
                _render_function(fn_name, fn_spec, indent=indent, full=full)
            )

    # Render classes
    for cls_name, cls_spec in spec.classes.items():
        is_cls_promoted = _is_class_promoted(cls_spec)
        effective_cls_visibility = cls_spec.visibility
        if cls_spec.visibility == "low" and is_cls_promoted:
            effective_cls_visibility = "medium"

        # A class is rendered if it's high-vis, or if it's medium-vis in a
        # high-vis container. A promoted class inside a promoted module also
        # needs a special check.
        if full or (
            effective_cls_visibility != "low"
            and (
                _should_render_member(effective_cls_visibility, effective_visibility)
                or is_cls_promoted
            )
        ):
            # Create a new spec with the correct effective visibility to pass down
            spec_to_render = dataclasses.replace(
                cls_spec, visibility=effective_cls_visibility
            )
            rendered_classes.append(
                _render_class(cls_name, spec_to_render, indent=indent, full=full)
            )

    rendered_classes.sort()
    rendered_fns.sort()
    rendered_members = rendered_classes + rendered_fns

    if not rendered_members:
        output.append(f"{indent}...")
    else:
        output.extend(rendered_members)

    return "\n".join(output)


def _render_function(
    name: str,
    spec: RegisteredFn,
    indent: str = "",
    is_method: bool = False,
    full: bool = False,
) -> str:
    """Renders a single function or method signature."""
    signature = inspect.signature(spec.fn)
    params = []
    for i, (p_name, p) in enumerate(signature.parameters.items()):
        if is_method and i == 0:
            params.append(p_name)  # self/cls
            continue

        param_str = p_name
        if p.annotation is not inspect.Parameter.empty:
            param_str += f": {p.annotation.__name__}"
        if p.default is not inspect.Parameter.empty:
            param_str += f" = {repr(p.default)}"
        params.append(param_str)

    return_str = ""
    if signature.return_annotation is not inspect.Signature.empty:
        return_str = f" -> {signature.return_annotation.__name__}"

    signature_line = f"{indent}def {name}({', '.join(params)}){return_str}:"

    # For medium visibility, just show that there's a body but no details.
    if not full and spec.visibility == "medium":
        return f"{signature_line}\n{indent}    ..."

    docstring = _render_docstring(spec.docstring, indent=indent + "    ", full=full)
    return f"{signature_line}\n{docstring}"


def _render_class(
    name: str, spec: RegisteredClass, indent: str = "", full: bool = False
) -> str:
    """Renders a single class definition based on its visibility."""
    member_indent = indent + "    "
    output = [f"{indent}class {name}:"]
    init_str = []
    attr_strs = []
    meth_strs = []

    # Render __init__ from constructor if available
    if spec.constructable and spec.visibility in ("high", "medium"):
        init_fn = spec.cls.__init__
        # Check if there's a specific override for __init__
        init_method_spec = spec.methods.get("__init__")
        doc = (
            init_method_spec.docstring
            if init_method_spec and init_method_spec.docstring is not None
            else init_fn.__doc__
        )

        init_fn_spec = RegisteredFn(fn=init_fn, docstring=doc, visibility="high")
        init_str.append(
            _render_function(
                "__init__",
                init_fn_spec,
                indent=member_indent,
                is_method=True,
                full=full,
            )
        )

    # For high- or medium-visibility classes, render all high-visibility members.
    # A medium-visibility class still needs to show its high-visibility children.
    if spec.visibility in ("high", "medium") or full:
        # Render attributes
        for attr_name, attr_spec in spec.attrs.items():
            if not full and attr_spec.visibility != "high":
                continue
            type_hint = ""
            if attr_name in spec.cls.__annotations__:
                type_hint = f": {spec.cls.__annotations__[attr_name].__name__}"
            attr_strs.append(f"{member_indent}{attr_name}{type_hint}")

        # Render methods
        for meth_name, meth_spec in spec.methods.items():
            if meth_name == "__init__":
                continue  # Already handled
            if not _should_render_member(
                meth_spec.visibility or spec.visibility, spec.visibility, full
            ):
                continue
            method = getattr(spec.cls, meth_name)
            doc = (
                meth_spec.docstring
                if meth_spec.docstring is not None
                else method.__doc__
            )
            meth_fn_spec = RegisteredFn(
                fn=method,
                docstring=doc,
                visibility=meth_spec.visibility or spec.visibility,
            )
            meth_strs.append(
                _render_function(
                    meth_name,
                    meth_fn_spec,
                    indent=member_indent,
                    is_method=True,
                    full=full,
                )
            )

    attr_strs.sort()
    meth_strs.sort()
    rendered_members = init_str + attr_strs + meth_strs

    if not rendered_members:
        # Only "class MyClass:" if nothing was rendered
        output.append(f"{member_indent}...")
    else:
        output.extend(rendered_members)

    return "\n".join(output)


def _render_docstring(doc: str | None, indent: str = "", full: bool = False) -> str:
    """Renders a formatted docstring."""
    if not doc:
        if full:
            return f'{indent}""""""'
        return f'{indent}"""..."""'

    clean_doc = inspect.cleandoc(doc)
    # Add indentation to each line
    indented_doc = "\n".join(f"{indent}{line}" for line in clean_doc.split("\n"))
    return f'{indent}"""\n{indented_doc}\n{indent}"""'
