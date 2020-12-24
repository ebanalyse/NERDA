# Helper function that flattens a list of lists
def flatten(xs):
    return [item for sublist in xs for item in sublist]

def tester(a, b):
    return a + b

def match_kwargs(function, **kwargs):
    
    arg_count = function.__code__.co_argcount
    args = function.__code__.co_varnames[:arg_count]

    args_dict = {}
    for k, v in kwargs.items():
        if k in args:
            args_dict[k] = v

    return args_dict