# Helper function that flattens a list of lists
def flatten(xs):
    return [item for sublist in xs for item in sublist]

# def match_args(function, **kwargs):
# 
#     arg_count = function.func_code.co_argcount
#     args = function.func_code.co_varnames[:arg_count]
# 
#     args_dict = {}
#     for k, v in **kwargs.iteritems():
#         if k in args:
#             args_dict[k] = v
# 
#     return v
