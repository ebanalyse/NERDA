from .preprocessing import create_dataloader
import torch
import numpy as np
from tqdm import tqdm 
from nltk.tokenize import sent_tokenize, word_tokenize

# NOTE: genbrug kode fra evaluering af model på validering?
# TODO: add batch_size, num_workers (til dataloader) som args.
def predict(network = None, 
            sentences = None,
            transformer_tokenizer = None,
            transformer_config = None,
            max_len = 128,
            device = None,
            tag_encoder = None):

    # set network to appropriate mode.
    network.eval()

    # fill 'dummy' tags.
    tag_fill = [tag_encoder.classes_[0]]
    tags_dummy = [tag_fill * len(sent) for sent in sentences]
    
    # TODO: kan vi genbruge fra validation?
    # TODO: kan vi reducere til danlp-logik?
    dl = create_dataloader(sentences,
                           tags_dummy, 
                           transformer_tokenizer,
                           transformer_config,
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

            # subset predictions for original word tokens
            preds = [prediction for prediction, offset in zip(preds.tolist(), dl.get('offsets')) if offset]
            # Remove special tokens ('CLS' + 'SEP')
            preds = preds[1:-1]  

            # make sure resulting predictions have same length as
            # original sentence.
            assert len(preds) == len(sentences[i])            

            predictions.append(preds)

    return predictions

def predict_text(network = None, 
                 text = "Ronaldo spiller i Juventus. Han er god.",
                 transformer_tokenizer = None,
                 transformer_config = None,
                 max_len = 128,
                 device = None,
                 tag_encoder = None,
                 sent_tokenizer = sent_tokenize,
                 word_tokenizer = word_tokenize):

    sentences = sent_tokenize(text)

    sentences = [word_tokenize(sentence) for sentence in sentences]

    predictions = predict(network = network, 
                          sentences = sentences,
                          transformer_tokenizer = transformer_tokenizer,
                          transformer_config = transformer_config,
                          max_len = max_len,
                          device = device,
                          tag_encoder = tag_encoder)

    return sentences, predictions



