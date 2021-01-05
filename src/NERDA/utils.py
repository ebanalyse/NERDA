# Helper function that flattens a list of lists
def flatten(xs):
    return [item for sublist in xs for item in sublist]

def match_kwargs(function: function, **kwargs: str) -> dict:
    """Matches Arguments with Function

    Match keywords arguments with a function.

    Args:
        function (function): Function to match arguments for.
        **kwargs (str): keyword arguments to match against.

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
