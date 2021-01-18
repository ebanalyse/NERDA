"""
This section covers functionality for computing performance
for [NERDA.models.NERDA][] models.
"""

from typing import List
from sklearn.metrics import precision_recall_fscore_support
import warnings

def flatten(l: list):
    """Flattens list"""
    return [item for sublist in l for item in sublist]


def compute_f1_scores(y_pred: List[List[str]], 
                      y_true: List[List[str]], 
                      labels: List[str],
                      **kwargs) -> list:
    """Compute F1 scores.
    
    Computes F1 Scores

    Args:
        y_pred (List): predicted values.
        y_true (List): observed/true values.
        labels (List): all possible tags.
        kwargs: all optional arguments for precision/recall function.

    Returns:
        list: resulting F1 scores.

    """  
    # check inputs.
    assert sum([len(t) < len(p) for t, p in zip(y_true, y_pred)]) == 0, "Length of predictions must not exceed length of observed values"

    # check, if some lengths of observed values exceed predicted values.
    n_exceeds = sum([len(t) > len(p) for t, p in zip(y_true, y_pred)])
    if n_exceeds > 0:
        warnings.warn(f'length of observed values exceeded lengths of predicted values in {n_exceeds} cases and were truncated. _Consider_ increasing max_len parameter for your model.')

    # truncate observed values dimensions to match predicted values,
    # this is needed if predictions have been truncated earlier in 
    # the flow.
    y_true = [t[:len(p)] for t, p in zip(y_true, y_pred)]
    
    y_pred = flatten(y_pred)
    y_true = flatten(y_true) 

    f1_scores = precision_recall_fscore_support(y_true = y_true,
                                                y_pred = y_pred,
                                                labels = labels,
                                                **kwargs) 

    return f1_scores                                                                
