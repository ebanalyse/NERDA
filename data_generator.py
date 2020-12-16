import pandas as pd
import numpy as np
import re
from danlp.datasets import DDT
from sklearn import preprocessing

ddt = DDT()
ddt_ne = ddt.load_as_simple_ner(predefined_splits= True)
tags = ['O', 'B-ORG', 'B-LOC', 'B-PER','B-MISC', 'I-PER', 'I-LOC', 'I-MISC', 'I-ORG']
encoder = preprocessing.LabelEncoder()
encoder_ = encoder.fit(tags)

def prepare_split(split) -> pd.DataFrame: 
    """Prepares tokens and tags for classification task.

    Uses a label encoder to encode the tags into integer values.

    Args:
        split (list): split consisting of tokens [0] and tags [1].

    Returns:
        pd.DataFrame: resulting tokens and tags.
    """
    df = pd.DataFrame({'words' : split[0],
                       'tags' : split[1]})

    df['tags'] = df['tags'].apply(encoder_.transform)

    return df

# TODO: Overvej at give ddt_ne som input
def get_dane_data_split(splits = ["train", "validate", "test"]):
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
    splits_out = [prepare_split(ddt_ne[splits_map.get(x)]) for x in splits]

    # if only one split, don't list results
    if len(splits_out) == 1:
        splits_out = splits_out[0]

    return splits_out
