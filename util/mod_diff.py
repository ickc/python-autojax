import inspect

from autojax import jax, numba, original


def get_callable_functions(module):
    """
    Extract all callable functions from a module, including decorated functions.

    Args:
        module: The module to extract functions from

    Returns:
        dict: A dictionary mapping function names to function objects
    """
    return {
        name: attr
        for name in dir(module)
        if callable(attr := getattr(module, name)) and not (name.startswith("_") or inspect.isclass(attr) or inspect.ismodule(attr))
    }


def diff_module(mod1, mod2):
    """Show the differences between 2 modules.

    Given each module, find all functions defined in that functions first, call it funcs1 and funcs2.
    Note that they have to be functions, filter out non-functions.
    Then create a list of functions where their names only exists in funcs1 but not in funcs2.
    This is funcs_diff: list[str]

    For the functions that is common between funcs1 and func2,
    inspect their signature, and create a list of them whenever they are different.
    This is funcs_sig_diff: dict[str, tuple[list[str], list[str]] where the key is the function name `func` and the values are the list of args from `func` in mod1 and mod2 respectively.
    """
    funcs1 = get_callable_functions(mod1)
    funcs2 = get_callable_functions(mod2)
    funcs1_set = set(funcs1)
    funcs2_set = set(funcs2)

    funcs_sig_diff = {}

    common_funcs = funcs1_set & funcs2_set

    for func_name in common_funcs:
        # Get function objects
        func1 = funcs1[func_name]
        func2 = funcs2[func_name]

        sig1 = inspect.signature(func1)
        sig2 = inspect.signature(func2)
        params1 = tuple(sig1.parameters)
        params2 = tuple(sig2.parameters)
        if params1 != params2:
            funcs_sig_diff[func_name] = (params1, params2)

    return funcs1_set - funcs2_set, funcs2_set - funcs1_set, funcs_sig_diff


def print_diff(mod1, mod2):
    diff1, diff2, diff_sig = diff_module(mod1, mod2)
    mod1_name = mod1.__name__.split(".")[-1]
    mod2_name = mod2.__name__.split(".")[-1]
    print("=" * 80)
    print(f"Functions in {mod1_name} but not in {mod2_name}")
    for func_name in sorted(diff1):
        print(f"  {func_name}")
    print("-" * 80)
    print(f"Functions in {mod2_name} but not in {mod1_name}")
    for func_name in sorted(diff2):
        print(f"  {func_name}")
    print("-" * 80)
    print("Functions with different signatures")
    for func_name, (params1, params2) in sorted(diff_sig.items()):
        print(f"  {func_name}")
        print(f"    {mod1_name}:")
        for param in params1:
            print(f"      {param}")
        print(f"    {mod2_name}:")
        for param in params2:
            print(f"      {param}")


def main():
    print("=" * 80)
    print("Functions in original:")
    for func_name in sorted(get_callable_functions(original).keys()):
        print(f"  {func_name}")
    print_diff(original, numba)
    print_diff(numba, jax)


if __name__ == "__main__":
    main()
