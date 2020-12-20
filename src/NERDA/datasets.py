from danlp.datasets import DDT
from typing import Union, List, Dict

ddt = DDT()
ddt_ne = ddt.load_as_simple_ner(predefined_splits= True)

# helper function
def prepare_split(split, limit = None):
    if limit is not None:
        split = [x[:limit] for x in split]    
    out = {'sentences': split[0], 'tags': split[1]}
    return out

# TODO: Overvej at give ddt_ne som input
def get_dane_data(splits: Union[str, List[str]] = ["train", "validate", "test"], limit: int = None) -> Union[Dict, List[Dict]]:
    """Get DaNE Data

    Loads one or more data splits from the DaNE ressource.

    Args:
        splits (Union[str, List[str]], optional): Choose which splits to load. You can choose from 'train', 'validate' or 'test'. Defaults to ["train", "validate", "test"].
        limit (int, optional): Limit the number of observations to be returned from a given split. Defaults to None.

    Returns:
        Union[Dict, List[Dict]]: Dictionary with word-tokenized 'sentences' and NER 'tags' in IOB format.
    """
    if isinstance(splits, str):
        splits = [splits]

    # TODO: assert that splits are inside splits map.

    # dictionary with indices of DaNE splits
    splits_map = {'train': 0, 'validate': 1, 'test': 2}
    
    # extract and prepare splits
    splits_out = [prepare_split(ddt_ne[splits_map.get(x)], limit = limit) for x in splits]

    # if only one split, don't list results
    if len(splits_out) == 1:
        splits_out = splits_out[0]

    return splits_out
