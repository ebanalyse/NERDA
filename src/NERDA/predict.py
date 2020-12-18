import torch
import numpy as np
import pandas as pd
import sklearn.metrics 
from tqdm import tqdm 
import warnings
import nltk

from .dataset import create_dataloader
from .model import NER_BERT

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
            df = None,
            bert_model_name = 'bert-base-multilingual-uncased',
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

    # TODO: bert_model_name skal arves fra modellen.
    dr, dl = create_dataloader(df, 
                               bert_model_name, 
                               max_len = max_len, 
                               batch_size = 1, 
                               tag_encoder = tag_encoder)

    predictions = []
    sentences = []
    with torch.no_grad():
        for i, dl in enumerate(dl): 

            # extract word tokenized sentence.
            # TODO: Maybe return words. Or decode.
            sentence = df['words'].iloc[i]

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
            assert len(preds) == len(sentence)            

            sentences.append(sentence)
            predictions.append(preds)

    return sentences, predictions

