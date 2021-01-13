from sklearn.metrics import precision_recall_fscore_support
import warnings

def flatten(l: list):
    """Flattens list"""
    return [item for sublist in l for item in sublist]


def compute_f1_scores(y_pred: list, 
                      y_true: list, 
                      labels: list,
                      **kwargs) -> list:
    """Compute F1 Scores

    Computes F1 Scores. 

    Args:
        y_pred (list): predicted values.
        y_true (list): observed/true values.
        labels (list): all possible tags.
        kwargs: all optional arguments for precision/recall function.

    Returns:
        list: resulting F1 scores.
    """
    assert sum([len(t) < len(p) for t, p in zip(y_true, y_pred)]) == 0, "Length of predictions must not exceed length of observed values"

    # check, if some lengths of observed values exceed predicted values.
    n_exceeds = sum([len(t) > len(p) for t, p in zip(y_true, y_pred)])
    if n_exceeds > 0:
        warnings.warn(f'length of observed values exceeded lengths of predicted values in {n_exceeds} cases and were truncated. Consider increasing max_len parameter for your model.')

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
