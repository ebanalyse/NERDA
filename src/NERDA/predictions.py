from .preprocessing import create_dataloader
import torch
import numpy as np
from tqdm import tqdm 
from nltk.tokenize import sent_tokenize, word_tokenize

# TODO: add batch_size, num_workers (til dataloader) som args.
def predict(network = None, 
            sentences = None,
            transformer_tokenizer = None,
            transformer_config = None,
            max_len = 128,
            device = None,
            tag_encoder = None,
            tag_outside = None,
            batch_size = 1,
            num_workers = 1):

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

            preds = tag_encoder.inverse_transform(
                    outputs.argmax(2).cpu().numpy().reshape(-1)
                )

            # subset predictions for original word tokens
            preds = [prediction for prediction, offset in zip(preds.tolist(), dl.get('offsets')) if offset]
            # Remove special tokens ('CLS' + 'SEP')
            preds = preds[1:-1]  

            # make sure resulting predictions have same length as
            # original sentence.
            assert len(preds) == len(sentences[i])            

            predictions.append(preds)

    return predictions

def predict_text(network, 
                 text,
                 transformer_tokenizer,
                 transformer_config,
                 max_len,
                 device,
                 tag_encoder,
                 tag_outside,
                 batch_size = 1,
                 num_workers = 1,
                 sent_tokenize = sent_tokenize,
                 word_tokenize = word_tokenize):

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

