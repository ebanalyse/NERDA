"""
This section covers functionality for computing predictions
with a [NERDA.models.NERDA][] model.
"""

from .preprocessing import create_dataloader
import torch
import numpy as np
from tqdm import tqdm 
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List, Callable
import transformers
import sklearn.preprocessing

def predict(network: torch.nn.Module, 
            sentences: List[List[str]],
            transformer_tokenizer: transformers.PreTrainedTokenizer,
            transformer_config: transformers.PretrainedConfig,
            max_len: int,
            device: str,
            tag_encoder: sklearn.preprocessing.LabelEncoder,
            tag_outside: str,
            batch_size: int = 8,
            num_workers: int = 1) -> List[List[str]]:
    """Compute predictions.

    Computes predictions for a list with word-tokenized sentences 
    with a `NERDA` model.

    Args:
        network (torch.nn.Module): Network.
        sentences (List[List[str]]): List of lists with word-tokenized
            sentences.
        transformer_tokenizer (transformers.PreTrainedTokenizer): 
            tokenizer for transformer model.
        transformer_config (transformers.PretrainedConfig): config
            for transformer model.
        max_len (int): Maximum length of sentence after applying 
            transformer tokenizer.
        device (str): Computational device.
        tag_encoder (sklearn.preprocessing.LabelEncoder): Encoder
            for Named-Entity tags.
        tag_outside (str): Special 'outside' NER tag.
        batch_size (int, optional): Batch Size for DataLoader. 
            Defaults to 8.
        num_workers (int, optional): Number of workers. Defaults
            to 1.

    Returns:
        List[List[str]]: List of lists with predicted Entity
        tags.
    """
    # make sure, that input has the correct format. 
    assert isinstance(sentences, list), "'sentences' must be a list of list of word-tokens"
    assert isinstance(sentences[0], list), "'sentences' must be a list of list of word-tokens"
    assert isinstance(sentences[0][0], str), "'sentences' must be a list of list of word-tokens"
    
    # set network to appropriate mode.
    network.eval()

    # fill 'dummy' tags (expected input for dataloader).
    tag_fill = [tag_encoder.classes_[0]]
    tags_dummy = [tag_fill * len(sent) for sent in sentences]
    
    dl = create_dataloader(sentences = sentences,
                           tags = tags_dummy, 
                           transformer_tokenizer = transformer_tokenizer,
                           transformer_config = transformer_config,
                           max_len = max_len, 
                           batch_size = batch_size, 
                           tag_encoder = tag_encoder,
                           tag_outside = tag_outside,
                           num_workers = num_workers)

    predictions = []
    
    with torch.no_grad():
        for i, dl in enumerate(dl): 

            outputs = network(**dl)   

            # conduct operations on sentence level.
            for i in range(outputs.shape[0]):
                
                # extract prediction and transform.
                preds = tag_encoder.inverse_transform(
                    outputs[i].argmax(-1).cpu().numpy()
                )

                # subset predictions for original word tokens.
                preds = [prediction for prediction, offset in zip(preds.tolist(), dl.get('offsets')[i]) if offset]
            
                # Remove special tokens ('CLS' + 'SEP').
                preds = preds[1:-1]
            
                # make sure resulting predictions have same length as
                # original sentence.
            
                # TODO: Move assert statement to unit tests. Does not work 
                # in boundary.
                # assert len(preds) == len(sentences[i])            

                predictions.append(preds)

    return predictions

def predict_text(network: torch.nn.Module, 
                 text: str,
                 transformer_tokenizer: transformers.PreTrainedTokenizer,
                 transformer_config: transformers.PretrainedConfig,
                 max_len: int,
                 device: str,
                 tag_encoder: sklearn.preprocessing.LabelEncoder,
                 tag_outside: str,
                 batch_size: int = 8,
                 num_workers: int = 1,
                 sent_tokenize: Callable = sent_tokenize,
                 word_tokenize: Callable = word_tokenize) -> tuple:
    """Compute Predictions for Text.

    Computes predictions for a text with `NERDA` model. 
    Text is tokenized into sentences before computing predictions.

    Args:
        network (torch.nn.Module): Network.
        text (str): text to predict entities in.
        transformer_tokenizer (transformers.PreTrainedTokenizer): 
            tokenizer for transformer model.
        transformer_config (transformers.PretrainedConfig): config
            for transformer model.
        max_len (int): Maximum length of sentence after applying 
            transformer tokenizer.
        device (str): Computational device.
        tag_encoder (sklearn.preprocessing.LabelEncoder): Encoder
            for Named-Entity tags.
        tag_outside (str): Special 'outside' NER tag.
        batch_size (int, optional): Batch Size for DataLoader. 
            Defaults to 8.
        num_workers (int, optional): Number of workers. Defaults
            to 1.

    Returns:
        tuple: sentence- and word-tokenized text with corresponding
        predicted named-entity tags.
    """
    assert isinstance(text, str), "'text' must be a string."
    sentences = sent_tokenize(text)

    sentences = [word_tokenize(sentence) for sentence in sentences]

    predictions = predict(network = network, 
                          sentences = sentences,
                          transformer_tokenizer = transformer_tokenizer,
                          transformer_config = transformer_config,
                          max_len = max_len,
                          device = device,
                          batch_size = batch_size,
                          num_workers = num_workers,
                          tag_encoder = tag_encoder,
                          tag_outside = tag_outside)

    return sentences, predictions

