import ast
from typing import Any, Callable

from .user_errors import AgexTypeError


def bind_arguments(
    func_name: str,
    func_args: ast.arguments,
    call_args: list[Any],
    call_kwargs: dict[str, Any],
    eval_fn: Callable[[ast.expr], Any] | None = None,
    evaluated_defaults: list[Any] | None = None,
    evaluated_kw_defaults: list[Any] | None = None,
) -> dict[str, Any]:
    """
    Binds arguments from a function call to the function's signature.

    This function is a simplified version of how Python's own argument
    binding works. It handles positional args, keyword args, default values,
    varargs (*args), and keyword-only args (**kwargs).

    Args:
        func_name: The name of the function being called (for error messages).
        func_args: The `ast.arguments` object from the function definition.
        call_args: A list of positional arguments from the call site.
        call_kwargs: A dictionary of keyword arguments from the call site.
        eval_fn: A function to evaluate the default value expressions. If not
            provided and evaluated_defaults is not provided, defaults cannot be evaluated.
        evaluated_defaults: Pre-evaluated default values (captured at definition time).
            When provided, these are used instead of re-evaluating AST nodes.
        evaluated_kw_defaults: Pre-evaluated keyword-only default values.

    Returns:
        A dictionary mapping argument names to their bound values.
    """
    bound_args = {}
    arg_names = [arg.arg for arg in func_args.args]
    num_positional_args = len(func_args.args)
    num_defaults = len(func_args.defaults)
    first_default_idx = num_positional_args - num_defaults

    # 1. Handle positional arguments
    if len(call_args) > num_positional_args and not func_args.vararg:
        raise AgexTypeError(
            f"{func_name}() takes {num_positional_args} positional arguments but {len(call_args)} were given"
        )

    for i, arg_name in enumerate(arg_names):
        if i < len(call_args):
            if arg_name in call_kwargs:
                raise AgexTypeError(
                    f"{func_name}() got multiple values for argument '{arg_name}'"
                )
            bound_args[arg_name] = call_args[i]
        elif arg_name in call_kwargs:
            bound_args[arg_name] = call_kwargs.pop(arg_name)
        elif i >= first_default_idx:
            default_idx = i - first_default_idx
            # Use pre-evaluated defaults if available (captures value at definition time)
            if evaluated_defaults is not None:
                bound_args[arg_name] = evaluated_defaults[default_idx]
            elif eval_fn:
                # Fallback: evaluate AST node (late binding - for backwards compat)
                default_val = func_args.defaults[default_idx]
                bound_args[arg_name] = eval_fn(default_val)
            else:
                raise AgexTypeError(
                    f"Cannot evaluate default argument for '{arg_name}' without an evaluator."
                )
        else:
            raise AgexTypeError(
                f"{func_name}() missing required positional argument: '{arg_name}'"
            )

    # 2. Handle vararg (*args)
    if func_args.vararg:
        bound_args[func_args.vararg.arg] = tuple(call_args[num_positional_args:])

    # 3. Handle keyword-only arguments
    for idx, arg in enumerate(func_args.kwonlyargs):
        arg_name = arg.arg
        if arg_name in call_kwargs:
            bound_args[arg_name] = call_kwargs.pop(arg_name)
        elif (
            idx < len(func_args.kw_defaults) and func_args.kw_defaults[idx] is not None
        ):
            # Use pre-evaluated kw_defaults if available
            if evaluated_kw_defaults is not None:
                bound_args[arg_name] = evaluated_kw_defaults[idx]
            elif eval_fn:
                # Fallback: evaluate AST node
                default_val = func_args.kw_defaults[idx]
                bound_args[arg_name] = eval_fn(default_val)
            else:
                raise AgexTypeError(
                    f"Cannot evaluate default keyword-only argument for '{arg_name}' without an evaluator."
                )
        else:
            raise AgexTypeError(
                f"{func_name}() missing required keyword-only argument: '{arg_name}'"
            )

    # 4. Handle kwarg (**kwargs)
    if func_args.kwarg:
        bound_args[func_args.kwarg.arg] = call_kwargs
    elif call_kwargs:
        unexpected_arg = next(iter(call_kwargs))
        raise AgexTypeError(
            f"{func_name}() got an unexpected keyword argument '{unexpected_arg}'"
        )

    return bound_args
