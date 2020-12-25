from danlp.datasets import DDT
from pathlib import Path
from typing import Union, List, Dict
import os

# helper function
def prepare_split(split, limit = None):
    if limit is not None:
        split = [x[:limit] for x in split]    
    out = {'sentences': split[0], 'tags': split[1]}
    return out

# TODO: Overvej at give ddt_ne som input
def get_dane_data(splits: Union[str, List[str]] = ["train", "validate", "test"], 
                  limit: int = None, 
                  cache_dir: str = os.path.join(str(Path.home()), '.NERDA')) -> Union[Dict, List[Dict]]:
    """Get DaNE Data

    Loads one or more data splits from the DaNE ressource.

    Args:
        splits (Union[str, List[str]], optional): Choose which splits to load. You can choose from 'train', 'validate' or 'test'. Defaults to ["train", "validate", "test"].
        limit (int, optional): Limit the number of observations to be returned from a given split. Defaults to None.
        cache_dir (str, optional): Directory where data is cached. Defaults to '.NERDA' folder in home directory.

    Returns:
        Union[Dict, List[Dict]]: Dictionary with word-tokenized 'sentences' and NER 'tags' in IOB format.
    """
    
    # (down)load DaNE data.
    # TODO: fix!
    # ddt = DDT(cache_dir = cache_dir)
    ddt = DDT()
    dane = ddt.load_as_simple_ner(predefined_splits = True)
    
    if isinstance(splits, str):
        splits = [splits]

    # TODO: assert that splits are inside splits map.
    # dictionary with indices of DaNE splits
    splits_map = {'train': 0, 'validate': 1, 'test': 2}
    
    # extract and prepare splits
    splits_out = [prepare_split(dane[splits_map.get(x)], limit = limit) for x in splits]

    # if only one split, don't list results
    if len(splits_out) == 1:
        splits_out = splits_out[0]

    return splits_out
