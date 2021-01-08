from sklearn.metrics import precision_recall_fscore_support

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

    y_pred = flatten(y_pred)
    y_true = flatten(y_true)

    f1_scores = precision_recall_fscore_support(y_true = y_true,
                                                y_pred = y_pred,
                                                labels = labels,
                                                **kwargs) 

    return f1_scores                                                                
