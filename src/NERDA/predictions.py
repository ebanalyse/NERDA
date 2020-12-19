from .preprocessing import create_dataloader
from .networks import NER_BERT
import torch
import numpy as np
import sklearn.metrics 
from tqdm import tqdm 
import warnings
import nltk


# Helper function that flattens a list of lists
def flatten(xs):
    return [item for sublist in xs for item in sublist]

# NOTE: er det ikke F1, der uddrages?
def compute_performance(predictions, 
                        targets, 
                        tag_scheme = [ 
                            'B-PER',
                            'I-PER', 
                            'I-ORG', 
                            'B-ORG', 
                            'B-LOC', 
                            'I-LOC', 
                            'B-MISC', 
                            'I-MISC'
                            ]):
    """
    Calculates and print out the accuracy scores for the DaNE test dataset.   
    """
    flat_preds = flatten(predictions)
    flat_labels = flatten(targets)

    # NOTE: skal tag_scheme gives eksplicit.
    scores_micro = sklearn.metrics.precision_recall_fscore_support(y_true = flat_labels,
                                                                   y_pred = flat_preds,
                                                                   labels = tag_scheme,
                                                                   average = 'micro' ) 

    scores_class = sklearn.metrics.precision_recall_fscore_support(y_true = flat_labels,
                                                                   y_pred = flat_preds,
                                                                   labels = tag_scheme,
                                                                   average = None ) 

    print('MICRO F1: ', scores_micro[2])
    # NOTE: igen
    class_scores = list(zip(tag_scheme, scores_class[2]))
    
    for score in class_scores:
        print(score[0], ': ' , score[1], '\n')

    return class_scores

# NOTE: genbrug kode fra evaluering af model på validering?
def predict(network = None, 
            sentences = None,
            transformer_tokenizer = None,
            max_len = 128,
            device = None,
            tag_encoder = None):

    """[summary]

    Args:
        df_test
        
    Returns:
        [List of strings]: returns the predictions for each of the sentences in the test_data provided.
    """

    # set network to appropriate mode.
    network.eval()

    # fill 'dummy' tags.
    tag_fill = [tag_encoder.classes_[0]]
    tags_dummy = [tag_fill * len(sent) for sent in sentences]
    
    # TODO: kan vi genbruge fra validation?
    # TODO: kan vi reducere til danlp-logik?
    dr, dl = create_dataloader(sentences,
                               tags_dummy, 
                               transformer_tokenizer,
                               max_len = max_len, 
                               batch_size = 1, 
                               tag_encoder = tag_encoder)

    predictions = []
    
    with torch.no_grad():
        for i, dl in enumerate(dl): 

            outputs = network(**dl)   

            preds = tag_encoder.inverse_transform(
                    outputs.argmax(2).cpu().numpy().reshape(-1)
                )

            # subset predictions for origional word tokens
            preds = [prediction for prediction, offset in zip(preds.tolist(), dl.get('offsets')) if offset]
            # Remove special tokens ('CLS' + 'SEP')
            preds = preds[1:-1]  

            # make sure resulting predictions have same length as
            # original sentence.
            assert len(preds) == len(sentences[i])            

            predictions.append(preds)

    return predictions

