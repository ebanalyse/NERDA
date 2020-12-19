from danlp.datasets import DDT

ddt = DDT()
ddt_ne = ddt.load_as_simple_ner(predefined_splits= True)

# helper function
def prepare_split(split, limit = None):
    if limit is not None:
        split = [x[:limit] for x in split]    
    out = {'sentences': split[0], 'tags': split[1]}
    return out

# TODO: Overvej at give ddt_ne som input
def get_dane_data(splits = ["train", "validate", "test"], limit = None):
    """
    Returns the train, validate and test datasets as dataframes.
    """

    if isinstance(splits, str):
        splits = [splits]

    # TODO: assert that splits are inside splits map.

    # dictionary with indices of DaNE splits
    splits_map = {'train': 0, 'validate': 1, 'test': 2}
    
    # extract and prepare splits.
    # TODO: results as dict would be better.
    splits_out = [prepare_split(ddt_ne[splits_map.get(x)], limit = limit) for x in splits]

    # if only one split, don't list results
    if len(splits_out) == 1:
        splits_out = splits_out[0]

    return splits_out
