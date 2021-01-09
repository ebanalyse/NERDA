from typing import Callable

def match_kwargs(function: Callable, **kwargs) -> dict:
    """Matches Arguments with Function

    Match keywords arguments with the arguments of a function.

    Args:
        function (function): Function to match arguments for.
        kwargs: keyword arguments to match against.

    Returns:
        dict: dictionary with matching arguments and their
        respective values.

    """
    arg_count = function.__code__.co_argcount
    args = function.__code__.co_varnames[:arg_count]

    args_dict = {}
    for k, v in kwargs.items():
        if k in args:
            args_dict[k] = v

    return args_dict
