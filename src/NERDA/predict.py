import torch
import numpy as np
import pandas as pd
import sklearn.metrics 
from tqdm import tqdm 
import warnings
import nltk

from .utils import target_tags
from .data_generator import encoder_, get_dane_data_split
from .dataset import create_dataloader
from .model import NER_BERT

# Helper function that flattens a list of lists
def flatten(xs):
    return [item for sublist in xs for item in sublist]

# NOTE: er det ikke F1, der uddrages?
def get_accuracy(predictions, targets):
    """
    Calculates and print out the accuracy scores for the DaNE test dataset.   
    """
    flat_preds = flatten(predictions)
    flat_labels = flatten(targets)

    # NOTE: utils.target_tags bør hentes fra modelobjekt/parametriseres.
    scores_micro = sklearn.metrics.precision_recall_fscore_support(y_true = flat_labels,
                                                                   y_pred= flat_preds,
                                                                   labels= target_tags,
                                                                   average= 'micro' ) 

    scores_class = sklearn.metrics.precision_recall_fscore_support(y_true = flat_labels,
                                                                   y_pred= flat_preds,
                                                                   labels= target_tags,
                                                                   average= None ) 

    print('MICRO F1: ', scores_micro[2])
    # NOTE: igen
    class_scores = list(zip(target_tags, scores_class[2]))
    for score in class_scores:
        print(score[0], ': ' , score[1], '\n')

# TODO: implementér som metode til modelobjekt. Skal alligevel ændres mhp. pipeline
# NOTE: adskil prædiktion fra performance measurement? Behov for yderligere modularisering.
# NOTE: genbrug kode fra evaluering af model på validering?
# TODO: Der mangler en funktion til at tage en rå streng (ny observation) og prædiktere den. Hent
# evt. inspiration fra 'danlp'
def predict(network = None, 
            df_test = None,
            run_dane_inference = True,
            bert_model_name = 'bert-base-multilingual-uncased',
            print_f1_scores = False,
            max_len = 128,
            device = None,
            model_path = "model.bin"):

    """[summary]

    Args:
        df_test
        
    Returns:
        [List of strings]: returns the predictions for each of the sentences in the test_data provided.
    """

    # NOTE: tokenizer bør arves fra den trænede model
    encoder = data_generator.encoder_

    # NOTE: selvstændig funktion??
    if run_dane_inference and df_test is not None:
        df_test = data_generator.get_dane_data_split('test')
        df_test = df_test.iloc[1:5]

    refs = df_test['tags'].apply(encoder.inverse_transform)

    dr_test, dl_test = create_dataloader(df_test, bert_model_name, max_len = 128, batch_size = 1)

    predictions = []
    sentences = []
    with torch.no_grad():
        for i, dl in enumerate(dl_test): 

            # extract word tokenized sentence.
            # TODO: Maybe return words. Or decode.
            sentence = df_test['words'].iloc[i]

            outputs = network(**dl)   

            preds = encoder.inverse_transform(
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

    if print_f1_scores:
        get_accuracy(predictions, refs.tolist())

    return sentences, predictions

if __name__ == '__main__' :
    predict(print_f1_scores = True, run_dane_inference = True)
    text = "Pernille Rosenkrantz-Theil kommer fra Vejle"
    import nltk
    words = nltk.word_tokenize(text)
    tags = [8] * len(words)
    import pandas as pd
    df = pd.DataFrame({'words': [words], 'tags': [tags]})
    sentences, predictions = predict(df_test = df)
    print(list(zip(sentences, predictions)))
    