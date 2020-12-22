from .preprocessing import create_dataloader
import torch
import numpy as np
from tqdm import tqdm 

# NOTE: genbrug kode fra evaluering af model på validering?
def predict(network = None, 
            sentences = None,
            transformer_tokenizer = None,
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

def tester(**kwargs):
    print(kwargs)

