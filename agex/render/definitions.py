import dataclasses
import importlib
import inspect
from types import ModuleType
from typing import Any

from agex.agent.base import BaseAgent
from agex.agent.policy.describe import (
    collect_class_candidate_names,
    describe_namespace,
    get_effective_member_spec,
)
from agex.agent.policy.resolve import make_predicate
from agex.agent.utils import get_instance_attributes_from_init

from ..agent.datatypes import (
    MemberSpec,
    RegisteredClass,
    RegisteredFn,
    RegisteredModule,
    RegisteredObject,
    Visibility,
)


def _is_sub_agent_function(fn: Any) -> bool:
    """Check if a function is a sub-agent function (TaskUserFunction)."""
    from agex.eval.functions import TaskUserFunction

    # Direct TaskUserFunction
    if isinstance(fn, TaskUserFunction):
        return True

    # Check if it's a wrapper around a TaskUserFunction (from registration.py)
    # When a TaskUserFunction is registered, it creates a wrapper that preserves metadata
    if hasattr(fn, "__name__") and hasattr(fn, "__doc__"):
        # The wrapper preserves the original function's docstring or sets it to "User-defined function"
        # We need to check if the underlying function is a TaskUserFunction
        # Look for the dual-decorator attributes that indicate a task function
        if hasattr(fn, "__agex_task_namespace__"):
            return True

    return False


def _render_type_annotation(
    annotation: Any, available_classes: set[str] | None = None
) -> str:
    """Renders a type annotation to a string, handling complex types."""
    if annotation is inspect.Parameter.empty or annotation is None:
        return ""

    # Get the string representation, which is usually good for complex types
    s = str(annotation)

    # Clean up common boilerplate for better readability
    s = s.replace("typing.", "")
    s = s.replace("<class '", "").replace("'>", "")

    # Clean up module prefixes for available classes
    if available_classes:
        for class_name in available_classes:
            # Replace patterns like __main__.ClassName with just ClassName
            s = s.replace(f"__main__.{class_name}", class_name)
            # Also handle other module patterns
            import re

            s = re.sub(rf"\b\w+\.{re.escape(class_name)}\b", class_name, s)

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
    # Collect available class names for type annotation rendering (from policy __main__)
    main_ns = agent._policy.namespaces.get("__main__")  # type: ignore[attr-defined]
    available_classes = set(main_ns.classes.keys()) if main_ns else set()

    # Render standalone functions from policy __main__
    if main_ns:
        for name, ms in main_ns.fns.items():
            eff_vis = ms.visibility or main_ns.visibility
            if not full and eff_vis != "high":
                continue
            fn_obj = main_ns.fn_objects.get(name)
            if not fn_obj:
                continue
            fn_spec = RegisteredFn(
                fn=fn_obj, docstring=ms.docstring or fn_obj.__doc__, visibility=eff_vis
            )
            output.append(
                _render_function(
                    name, fn_spec, available_classes=available_classes, full=full
                )
            )

    # Render classes using policy-backed adapter for __main__ virtual namespace
    classes_to_render = []
    if main_ns:
        for name, rc in main_ns.classes.items():  # type: ignore[union-attr]
            adapted = _policy_main_class_to_registered_class(agent, rc.cls)
            if not adapted:
                continue
            is_promoted = _is_class_promoted(adapted)
            effective_visibility = adapted.visibility
            if adapted.visibility == "low" and is_promoted:
                effective_visibility = "medium"

            if full or effective_visibility != "low":
                spec_to_render = dataclasses.replace(
                    adapted, visibility=effective_visibility
                )
                classes_to_render.append(
                    _render_class(
                        name,
                        spec_to_render,
                        available_classes=available_classes,
                        full=full,
                    )
                )

    # Add helpful header if classes are present
    if classes_to_render:
        output.append("# Available classes (use directly, no import needed):")
        output.extend(classes_to_render)

    # Render modules with helpful header (adapter: prefer policy; fallback to legacy)
    modules_to_render = []
    # Prefer policy namespaces first
    for ns_name, ns in agent._policy.namespaces.items():  # type: ignore[attr-defined]
        if ns.kind != "module":
            continue
        adapted = _policy_namespace_to_registered_module(agent, ns_name)
        if adapted:
            rendered = _render_module(ns_name, adapted, full=full)
            if rendered:
                modules_to_render.append(rendered)
    # No legacy fallback: modules are enumerated from policy only

    if modules_to_render:
        output.append("# Available modules (import before using):")
        output.extend(modules_to_render)
        note = "# To use the modules above, import them as you would any Python module:"
        note += "\n# - import some_module"
        note += "\n# - import some_module as alias"
        note += "\n# Note: you may not have full access to these modules or classes."
        note += "\n# If you do not, you will see this as an error in your stdout (such as a 'no attribute' error)."
        output.append(note)

    # Render registered objects (live objects) from policy instance namespaces
    for ns_name, ns in agent._policy.namespaces.items():  # type: ignore[attr-defined]
        if ns.kind != "instance":
            continue
        # Describe instance namespace to honor include/exclude and configure
        include_pred = make_predicate(ns.include)
        exclude_pred = make_predicate(ns.exclude)
        methods: dict[str, MemberSpec] = {}
        properties: dict[str, MemberSpec] = {}
        # Use live object if present to distinguish callables
        live_obj = getattr(ns, "obj", None)
        if live_obj is not None:
            for attr in dir(live_obj):
                if attr.startswith("_"):
                    continue
                if not (include_pred(attr) and not exclude_pred(attr)):
                    continue
                cfg = ns.configure.get(attr, MemberSpec())
                vis = cfg.visibility or ns.visibility
                doc = cfg.docstring
                try:
                    member = getattr(live_obj, attr)
                except Exception:
                    member = None
                if callable(member):
                    methods[attr] = MemberSpec(visibility=vis, docstring=doc)
                else:
                    properties[attr] = MemberSpec(visibility=vis, docstring=doc)
        spec = RegisteredObject(
            name=ns_name,
            visibility=ns.visibility,
            methods=methods,
            properties=properties,
        )
        rendered_object = _render_object(ns_name, spec, agent, full=full)
        if rendered_object:
            output.append(rendered_object)

    return "\n\n".join(output)


def _policy_namespace_to_registered_module(
    agent: BaseAgent, ns_name: str
) -> RegisteredModule | None:
    """
    Build a RegisteredModule-equivalent from a policy module namespace using
    describe() logic. This is a compatibility adapter to allow the existing
    rendering pipeline to operate without changes.
    """
    ns = agent._policy.namespaces.get(ns_name)  # type: ignore[attr-defined]
    if not ns or ns.kind != "module":
        return None
    mod = ns._ensure_module_loaded()
    top = describe_namespace(ns, include_low=True)

    mod_fns: dict[str, MemberSpec] = {}
    mod_consts: dict[str, MemberSpec] = {}
    mod_classes: dict[str, RegisteredClass] = {}

    for member_name, desc in top.items():
        if desc.kind == "fn":
            mod_fns[member_name] = MemberSpec(
                visibility=desc.visibility, docstring=desc.docstring
            )
        elif desc.kind == "class":
            cls_obj = getattr(mod, member_name, None)
            if cls_obj is None:
                continue
            # Build class members using shared helpers with dotted precedence
            include_pred = make_predicate(ns.include)
            exclude_pred = make_predicate(ns.exclude)
            cls_attrs: dict[str, MemberSpec] = {}
            cls_methods: dict[str, MemberSpec] = {}

            # Determine constructability from class-level config, default True
            cls_cfg = ns.configure.get(member_name, MemberSpec())
            cls_is_constructable = (
                cls_cfg.constructable if cls_cfg.constructable is not None else True
            )

            candidates = collect_class_candidate_names(
                cls_obj, ns=ns, constructable=cls_is_constructable
            )

            # Apply include/exclude on dotted names
            selected = set()
            for short in candidates:
                dotted = f"{member_name}.{short}"
                if include_pred(dotted) and not exclude_pred(dotted):
                    selected.add(short)

            for short in selected:
                obj = getattr(cls_obj, short, None)
                cm_cfg = get_effective_member_spec(
                    ns, class_name=member_name, member_name=short
                )
                cm_vis: Visibility | None = cm_cfg.visibility or (
                    cls_cfg.visibility or ns.visibility
                )
                cm_doc = cm_cfg.docstring

                if inspect.isroutine(obj) or short == "__init__":
                    if short == "__init__" and cm_cfg.visibility is None:
                        cm_vis = "medium"
                    cls_methods[short] = MemberSpec(visibility=cm_vis, docstring=cm_doc)
                else:
                    cls_attrs[short] = MemberSpec(visibility=cm_vis, docstring=cm_doc)

            reg_cls = RegisteredClass(
                cls=cls_obj,
                visibility=cls_cfg.visibility or ns.visibility,
                constructable=cls_is_constructable,
                attrs=cls_attrs,
                methods=cls_methods,
            )
            mod_classes[member_name] = reg_cls
        else:
            mod_consts[member_name] = MemberSpec(
                visibility=desc.visibility, docstring=desc.docstring
            )

    # Targeted support for dotted members explicitly configured on recursive modules.
    # We add functions/classes like "routing.shortest_path" when their effective
    # visibility is medium/high. Skip class-member dotted keys (e.g., "Cls.meth").
    def _resolve_dotted_member(root: ModuleType, dotted: str) -> Any | None:
        parts = dotted.split(".")
        current: Any = root
        for idx, part in enumerate(parts):
            try:
                if hasattr(current, part):
                    current = getattr(current, part)
                    continue
            except Exception:
                return None
            # Attempt to import a submodule when attribute is missing
            if isinstance(current, ModuleType):
                base = current.__name__
                try:
                    current = importlib.import_module(f"{base}.{part}")
                    continue
                except Exception:
                    return None
            return None
        return current

    # Only attempt when module namespace is recursive
    if getattr(ns, "recursive", False):
        # Prefer explicitly configured dotted keys
        for dotted_key, ms in ns.configure.items():
            if "." not in dotted_key:
                continue
            # Avoid class-member dotted keys by skipping if the left part is a class
            left, _sep, _rest = dotted_key.partition(".")
            try:
                left_obj = getattr(mod, left)
                if inspect.isclass(left_obj):
                    continue
            except Exception:
                # If left attr missing on root, still allow resolution via import
                pass

            eff_vis = ms.visibility or ns.visibility
            if eff_vis == "low":
                continue

            obj = _resolve_dotted_member(mod, dotted_key)
            if obj is None:
                continue
            # Capture docstring override or object's own
            doc = ms.docstring or getattr(obj, "__doc__", None)
            if inspect.isclass(obj):
                reg_cls = RegisteredClass(
                    cls=obj,
                    visibility=eff_vis,
                    constructable=True,
                    attrs={},
                    methods={},
                )
                mod_classes[dotted_key] = reg_cls
            elif inspect.isroutine(obj):
                # Store as a function member with the dotted name
                mod_fns[dotted_key] = MemberSpec(visibility=eff_vis, docstring=doc)
            else:
                mod_consts[dotted_key] = MemberSpec(visibility=eff_vis, docstring=doc)

    return RegisteredModule(
        name=ns_name,
        module=mod,
        visibility=ns.visibility,
        fns=mod_fns,
        consts=mod_consts,
        classes=mod_classes,
    )


def _policy_main_class_to_registered_class(
    agent: BaseAgent, py_cls: type
) -> RegisteredClass | None:
    """
    Adapt a class registered in the policy virtual main namespace into a
    RegisteredClass that matches legacy rendering behavior.
    """
    # Retrieve the per-class namespace spec captured at registration time.
    ns = getattr(agent._policy, "_class_namespaces", {}).get(py_cls)  # type: ignore[attr-defined]
    if ns is None:
        return None

    cls_name = getattr(py_cls, "__name__", "")

    # Determine class-level visibility with override precedence
    class_cfg = ns.configure.get(cls_name, MemberSpec())
    class_visibility = class_cfg.visibility or ns.visibility

    # Determine constructability
    try:
        rc = agent._policy.resolve_class_spec(py_cls)  # type: ignore[attr-defined]
        constructable = bool(rc.constructable) if rc is not None else True
    except Exception:
        constructable = True

    # Build include/exclude predicates
    include_pred = make_predicate(ns.include)
    exclude_pred = make_predicate(ns.exclude)

    # Generate candidate member names
    all_members: set[str] = set()
    for name, member in inspect.getmembers(py_cls):
        if not name.startswith("__") or name == "__init__":
            all_members.add(name)
    if hasattr(py_cls, "__annotations__"):
        all_members.update(py_cls.__annotations__.keys())

    # Include instance attributes from __init__ when wildcard patterns or explicit includes are used
    if (
        ns.include == "*"
        or (isinstance(ns.include, str) and "*" in ns.include)
        or isinstance(ns.include, (list, set, tuple))
    ):
        try:
            instance_attrs = get_instance_attributes_from_init(py_cls)
            all_members.update(instance_attrs)
        except Exception:
            pass

    # If include is an iterable of explicit names, add them to candidates too
    if isinstance(ns.include, (list, set, tuple)):
        for item in ns.include:
            if isinstance(item, str):
                all_members.add(item)

    # Select members via include/exclude
    selected_names = {n for n in all_members if include_pred(n) and not exclude_pred(n)}

    # Enforce constructability on __init__
    if constructable:
        selected_names.add("__init__")
    else:
        selected_names.discard("__init__")

    attrs: dict[str, MemberSpec] = {}
    methods: dict[str, MemberSpec] = {}

    for member_name in selected_names:
        obj = getattr(py_cls, member_name, None)

        # Look up config with both dotted and plain keys for override precedence
        dotted_key = f"{cls_name}.{member_name}"
        cfg = ns.configure.get(dotted_key, ns.configure.get(member_name, MemberSpec()))
        vis = cfg.visibility or class_visibility
        doc = cfg.docstring

        if member_name == "__init__" or inspect.isroutine(obj):
            methods[member_name] = MemberSpec(visibility=vis, docstring=doc)
        else:
            attrs[member_name] = MemberSpec(visibility=vis, docstring=doc)

    return RegisteredClass(
        cls=py_cls,
        visibility=class_visibility,
        constructable=constructable,
        attrs=attrs,
        methods=methods,
    )


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

    # Helper to resolve dotted attributes under the module when needed
    def _resolve_under_module(module: ModuleType, dotted: str) -> Any | None:
        if "." not in dotted:
            try:
                return getattr(module, dotted)
            except Exception:
                return None
        parts = dotted.split(".")
        current: Any = module
        for idx, part in enumerate(parts):
            try:
                if hasattr(current, part):
                    current = getattr(current, part)
                    continue
            except Exception:
                return None
            if isinstance(current, ModuleType):
                base = current.__name__
                try:
                    current = importlib.import_module(f"{base}.{part}")
                    continue
                except Exception:
                    return None
            else:
                return None
        return current

    # Render functions
    for fn_name, fn_member_spec in spec.fns.items():
        if _should_render_member(
            fn_member_spec.visibility or spec.visibility,
            effective_visibility,
            full=full,
        ):
            fn = _resolve_under_module(spec.module, fn_name)
            if fn is None:
                # Fallback placeholder if resolution fails
                fn = lambda: None
            doc = (
                fn_member_spec.docstring
                if fn_member_spec.docstring is not None
                else getattr(fn, "__doc__", None)
            )
            # We pass the member's own visibility down so the function renderer
            # knows whether to render the docstring or not.
            fn_spec = RegisteredFn(
                fn=fn,
                docstring=doc,
                visibility=fn_member_spec.visibility or spec.visibility,
            )
            rendered_fns.append(
                _render_function(
                    fn_name, fn_spec, indent=indent, available_classes=set(), full=full
                )
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
                _render_class(
                    cls_name,
                    spec_to_render,
                    indent=indent,
                    available_classes=set(),
                    full=full,
                )
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


def _render_object(
    name: str, spec: RegisteredObject, agent: BaseAgent, full: bool = False
) -> str:
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
            # Get the actual method from the agent's host object registry
            live_obj = agent._host_object_registry.get(name)
            if live_obj and hasattr(live_obj, method_name):
                # Use the actual method for proper signature rendering
                actual_method = getattr(live_obj, method_name)
                # Use the spec's docstring if provided, otherwise the method's own docstring
                doc = (
                    method_spec.docstring
                    if method_spec.docstring is not None
                    else actual_method.__doc__
                )
                method_fn_spec = RegisteredFn(
                    fn=actual_method,
                    docstring=doc,
                    visibility=method_spec.visibility or spec.visibility,
                )
            else:
                # Fallback to placeholder if the live object isn't available
                method_fn_spec = RegisteredFn(
                    fn=lambda: None,  # Placeholder function
                    docstring=method_spec.docstring,
                    visibility=method_spec.visibility or spec.visibility,
                )

            rendered_methods.append(
                _render_function(
                    method_name,
                    method_fn_spec,
                    indent=indent,
                    is_method=True,
                    available_classes=set(),
                    full=full,
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
    available_classes: set[str] | None = None,
    full: bool = False,
) -> str:
    """Renders a single function or method signature."""
    prefix = indent
    fn = spec.fn

    # Check if this is a sub-agent function
    is_sub_agent = _is_sub_agent_function(fn)

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
        signature_line = f"{prefix}{fn_name}(...):"

        # Add sub-agent comment if needed
        if is_sub_agent:
            signature_line += "  # Sub-agent task function"

        # Still try to show docstring if available and visibility is high or full mode
        docstring = spec.docstring or inspect.getdoc(fn)
        if (full or spec.visibility == "high") and docstring:
            return f"{signature_line}\n{_render_docstring(docstring, indent=indent + '    ', full=full)}"

        # In all other cases, the body is elided.
        return f"{signature_line}\n{indent}    ..."

    params = []
    for i, (p_name, p) in enumerate(signature.parameters.items()):
        # For unbound methods, skip the first parameter (self/cls) but add it without type annotation
        # For bound methods, the self parameter is already stripped by Python's introspection
        if is_method and i == 0 and not hasattr(fn, "__self__"):
            params.append(p_name)  # self/cls
            continue

        # Hide framework-injected parameters for sub-agent task functions
        if is_sub_agent and p_name in ("state", "on_event"):
            continue

        param_str = p_name
        type_str = _render_type_annotation(p.annotation, available_classes)
        if type_str:
            param_str += f": {type_str}"

        if p.default is not inspect.Parameter.empty:
            param_str += f" = {repr(p.default)}"
        params.append(param_str)

    return_str = ""
    ret_type_str = _render_type_annotation(
        signature.return_annotation, available_classes
    )
    if ret_type_str:
        return_str = f" -> {ret_type_str}"

    signature_line = f"{prefix}{fn_name}({', '.join(params)}){return_str}:"

    # Add sub-agent comment if needed
    if is_sub_agent:
        signature_line += "  # Sub-agent task function"

    docstring = spec.docstring or inspect.getdoc(fn)

    # Replace generic dataclass docstring with something more helpful
    if docstring == "Initialize self.  See help(type(self)) for accurate signature.":
        docstring = "Creates new instance"

    # Functions with high visibility (or when `full=True`) show their docstring.
    if (full or spec.visibility == "high") and docstring:
        return f"{signature_line}\n{_render_docstring(docstring, indent=indent + '    ', full=full)}"

    # In all other cases, the body is elided.
    return f"{signature_line}\n{indent}    ..."


def _render_class(
    name: str,
    spec: RegisteredClass,
    indent: str = "",
    available_classes: set[str] | None = None,
    full: bool = False,
) -> str:
    """Renders a single class definition based on its visibility."""
    member_indent = indent + "    "
    output = [f"{indent}class {name}:"]
    # Render class-level docstring for high visibility (or in full mode)
    if full or spec.visibility == "high":
        cls_doc = getattr(spec.cls, "__doc__", None)
        if cls_doc:
            output.append(_render_docstring(cls_doc, indent=member_indent, full=full))
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
            # Replace generic dataclass docstring with something more helpful
            if doc == "Initialize self.  See help(type(self)) for accurate signature.":
                doc = "Creates new instance"

        init_fn_spec = RegisteredFn(fn=init_fn, docstring=doc, visibility=vis)
        init_str.append(
            _render_function(
                "__init__",
                init_fn_spec,
                indent=member_indent,
                is_method=True,
                available_classes=available_classes,
                full=full,
            )
        )

    # For high- or medium-visibility classes, render members based on their visibility.
    if spec.visibility in ("high", "medium") or full:
        # Build a mapping of attribute names to their type annotations
        attr_type_hints = {}

        # First, check class-level annotations
        if hasattr(spec.cls, "__annotations__"):
            attr_type_hints.update(spec.cls.__annotations__)

        # Then, check __init__ method parameters for instance attributes
        if hasattr(spec.cls, "__init__"):
            try:
                init_signature = inspect.signature(spec.cls.__init__)
                for param_name, param in init_signature.parameters.items():
                    if (
                        param_name != "self"
                        and param.annotation != inspect.Parameter.empty
                    ):
                        # Map parameter name to attribute name (they should match)
                        attr_type_hints[param_name] = param.annotation
            except (ValueError, TypeError):
                # If signature inspection fails, continue without __init__ annotations
                pass

        # Render attributes
        for attr_name, attr_spec in spec.attrs.items():
            if not _should_render_member(
                attr_spec.visibility or spec.visibility, spec.visibility, full
            ):
                continue
            type_hint = ""
            if attr_name in attr_type_hints:
                type_hint = f": {_render_type_annotation(attr_type_hints[attr_name], available_classes)}"
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
                    available_classes=available_classes,
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
