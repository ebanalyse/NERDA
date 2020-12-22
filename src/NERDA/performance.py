from sklearn.metrics import precision_recall_fscore_support

# Helper function that flattens a list of lists
def flatten(xs):
    return [item for sublist in xs for item in sublist]

# NOTE: er det ikke F1, der uddrages?
def compute_f1_scores(y_pred, 
                      y_true, 
                      labels,
                      **kwargs):
    """
    Calculates and print out the accuracy scores for the DaNE test dataset.   
    """
    y_pred = flatten(y_pred)
    y_true = flatten(y_true)

    f1_scores = precision_recall_fscore_support(y_true = y_true,
                                                y_pred = y_pred,
                                                labels = labels,
                                                **kwargs) 

    return f1_scores                                                                
