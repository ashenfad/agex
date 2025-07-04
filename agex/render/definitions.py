import dataclasses
import inspect
from typing import Any

from agex.agent.base import BaseAgent

from ..agent.datatypes import (
    RegisteredClass,
    RegisteredFn,
    RegisteredModule,
    RegisteredObject,
    Visibility,
)


def _render_type_annotation(annotation: Any) -> str:
    """Renders a type annotation to a string, handling complex types."""
    if annotation is inspect.Parameter.empty or annotation is None:
        return ""

    # Get the string representation, which is usually good for complex types
    s = str(annotation)

    # Clean up common boilerplate for better readability
    s = s.replace("typing.", "")
    s = s.replace("<class '", "").replace("'>", "")

    return s


def render_definitions(agent: BaseAgent, full: bool = False) -> str:
    """
    Renders the registered functions, classes, and modules of an agent
    into a Python-like string of signatures and docstrings.

    The rendering is controlled by a visibility system:
    - `high`: Renders the full function signature and its docstring. If no
      docstring exists, the body will be `pass`.
    - `medium`: Renders only the function signature. The body is always `...`
      to indicate the implementation is hidden. The docstring is never shown.
    - `low`: The item is not rendered at all.

    If `full` is True, all members are rendered at their highest effective
    visibility, ignoring these rules.
    """
    output = []
    # Render standalone functions
    for name, spec in agent.fn_registry.items():
        if not full and spec.visibility != "high":
            continue
        output.append(_render_function(name, spec, full=full))

    # Render classes
    classes_to_render = []
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
            classes_to_render.append(_render_class(name, spec_to_render, full=full))

    # Add helpful header if classes are present
    if classes_to_render:
        output.append("# Available classes (use directly, no import needed):")
        output.extend(classes_to_render)

    # Render modules with helpful header
    modules_to_render = []
    for name, spec in agent.importable_modules.items():
        rendered_module = _render_module(name, spec, full=full)
        if rendered_module:
            modules_to_render.append(rendered_module)

    if modules_to_render:
        output.append("# Available modules (import before using):")
        output.extend(modules_to_render)

    # Render registered objects (live objects)
    for name, spec in agent.object_registry.items():
        rendered_object = _render_object(name, spec, full=full)
        if rendered_object:
            output.append(rendered_object)

    return "\n\n".join(output)


def _should_render_member(
    member_vis: Visibility, container_vis: Visibility, full: bool = False
) -> bool:
    """Determines if a member should be rendered based on its and its container's visibility."""
    if full:
        return True
    if member_vis == "high":
        return True
    if member_vis == "medium":
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

    # If a module is low-vis and not promoted, just show that it exists.
    if not full and effective_visibility == "low":
        return f"module {name}:\n    ..."

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


def _is_object_promoted(spec: RegisteredObject) -> bool:
    """An object is promoted if it contains any high-visibility methods or properties."""
    return any(m.visibility == "high" for m in spec.methods.values()) or any(
        p.visibility == "high" for p in spec.properties.values()
    )


def _render_object(name: str, spec: RegisteredObject, full: bool = False) -> str:
    """Renders a single registered object definition based on its visibility and its members' visibilities."""
    # Determine the object's effective visibility. A low-vis object can be
    # "promoted" to medium-vis if it contains a high-vis member.
    is_promoted = _is_object_promoted(spec)
    effective_visibility = spec.visibility
    if spec.visibility == "low" and is_promoted:
        effective_visibility = "medium"

    # If an object is low-vis and not promoted, just show that it exists.
    if not full and effective_visibility == "low":
        return f"object {name}:\n    ..."

    output = [f"object {name}:"]
    indent = "    "
    rendered_methods = []
    rendered_properties = []

    # Render methods
    for method_name, method_spec in spec.methods.items():
        if _should_render_member(
            method_spec.visibility or spec.visibility,
            effective_visibility,
            full=full,
        ):
            # Create a fake RegisteredFn for rendering consistency
            fake_fn_spec = RegisteredFn(
                fn=lambda: None,  # Placeholder function
                docstring=method_spec.docstring,
                visibility=method_spec.visibility or spec.visibility,
            )
            rendered_methods.append(
                _render_function(
                    method_name, fake_fn_spec, indent=indent, is_method=True, full=full
                )
            )

    # Render properties
    for prop_name, prop_spec in spec.properties.items():
        if _should_render_member(
            prop_spec.visibility or spec.visibility,
            effective_visibility,
            full=full,
        ):
            prop_line = f"{indent}{prop_name}: ..."
            if (full or prop_spec.visibility == "high") and prop_spec.docstring:
                prop_line += f"\n{_render_docstring(prop_spec.docstring, indent=indent + '    ', full=full)}"
            rendered_properties.append(prop_line)

    rendered_methods.sort()
    rendered_properties.sort()
    rendered_members = rendered_methods + rendered_properties

    if not rendered_members:
        output.append(f"{indent}...")
    else:
        output.extend(rendered_members)

    return "\n".join(output)


def _render_function(
    fn_name: str,
    spec: RegisteredFn,
    indent: str = "",
    is_method=False,
    full: bool = False,
) -> str:
    """Renders a single function or method signature."""
    prefix = indent
    fn = spec.fn
    try:
        signature = inspect.signature(fn)
    except ValueError:
        signature = None  # Fallback for builtins with no signature

    if inspect.iscoroutinefunction(fn) or inspect.iscoroutine(fn):
        prefix += "async def "
    else:
        prefix += "def "

    if signature is None:
        # For builtins that fail inspection, provide a fallback
        signature_line = f"{prefix}{fn_name}(...)"
        if not full and spec.visibility == "medium":
            return f"{signature_line}:"
        return signature_line

    params = []
    for i, (p_name, p) in enumerate(signature.parameters.items()):
        if is_method and i == 0:
            params.append(p_name)  # self/cls
            continue

        param_str = p_name
        type_str = _render_type_annotation(p.annotation)
        if type_str:
            param_str += f": {type_str}"

        if p.default is not inspect.Parameter.empty:
            param_str += f" = {repr(p.default)}"
        params.append(param_str)

    return_str = ""
    ret_type_str = _render_type_annotation(signature.return_annotation)
    if ret_type_str:
        return_str = f" -> {ret_type_str}"

    signature_line = f"{prefix}{fn_name}({', '.join(params)}){return_str}"

    docstring = spec.docstring or inspect.getdoc(fn)

    # Functions with high visibility (or when `full=True`) show their docstring.
    if (full or spec.visibility == "high") and docstring:
        return f"{signature_line}:\n{_render_docstring(docstring, indent=indent + '    ', full=full)}"

    # In all other cases, the body is elided.
    return f"{signature_line}:\n{indent}    ..."


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
        init_method_spec = spec.methods.get("__init__")

        doc = None
        vis = spec.visibility  # Default to class visibility
        if init_method_spec:
            doc = init_method_spec.docstring
            if init_method_spec.visibility:
                vis = init_method_spec.visibility

        # If there's no explicit doc override, use the function's own docstring.
        if doc is None:
            doc = init_fn.__doc__

        init_fn_spec = RegisteredFn(fn=init_fn, docstring=doc, visibility=vis)
        init_str.append(
            _render_function(
                "__init__",
                init_fn_spec,
                indent=member_indent,
                is_method=True,
                full=full,
            )
        )

    # For high- or medium-visibility classes, render members based on their visibility.
    if spec.visibility in ("high", "medium") or full:
        # Render attributes
        for attr_name, attr_spec in spec.attrs.items():
            if not _should_render_member(
                attr_spec.visibility or spec.visibility, spec.visibility, full
            ):
                continue
            type_hint = ""
            if (
                hasattr(spec.cls, "__annotations__")
                and attr_name in spec.cls.__annotations__
            ):
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
            # Use the spec's docstring if provided, otherwise the method's own.
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
        output.append(f"{member_indent}pass")
    else:
        output.extend(rendered_members)

    return "\n".join(output)


def _render_docstring(doc: str | None, indent: str = "", full: bool = False) -> str:
    """Renders a formatted docstring."""
    if not doc:
        if full:
            return f'{indent}""""""'
        return ""

    clean_doc = inspect.cleandoc(doc)
    # Add indentation to each line
    indented_doc = "\n".join(f"{indent}{line}" for line in clean_doc.split("\n"))
    return f'{indent}"""\n{indented_doc}\n{indent}"""'
