import pandas as pd
import numpy as np
import torch
from .utils import enforce_reproducibility
from .dataset import create_dataloader
from .model_functions import train_, validate_
from .data_generator import get_dane_data_split
import torch

from .model import NER_BERT
from sklearn import preprocessing
from sklearn import model_selection
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

def train_model(network,
                df_train = get_dane_data_split('train'), 
                df_validate = get_dane_data_split('validate'), 
                bert_model_name = 'bert-base-multilingual-uncased',
                max_len = 128,
                train_batch_size = 16,
                validation_batch_size = 8,
                epochs = 5,
                warmup_steps = 0,
                custom_weight_decay = False,
                learning_rate = 5e-5,
                return_valid_loss = False,
                device = None,
                fixed_seed = 42):
    
    """
    Runs the fine-tuning for the BERT based NER model. 
    
    Calling the function with no defined parameters results in a fine-tuning on the DaNE data, with a basic
    set of hyperparameters. 

    Args:
        df_train ([type]): [description]
        df_validate ([type]): [description]
        bert_model_name ([type]): [description]. Defaults to utils.mbert
        use_dane_data (bool): [description]. Defaults to True.
        max_len (int): [description]. Defaults to 128.
        train_batch_size (int): [description]. Defaults to 16.
        validation_batch_size (int): [description]. Defaults to 8.
        epochs (int): [description]. Defaults to 5.
        warmup_steps (int): [description]. Defaults to 0.
        custom_weight_decay (bool): [description]. Defaults to False.
        learning_rate ([type]): [description]. Defaults to 5e-5.
    """
    if fixed_seed is not None:
        enforce_reproducibility(fixed_seed)
    
    # prepare datasets for modelling by creating data readers and loaders
    # TODO: parametrize num_workers.
    dr_train, dl_train = create_dataloader(df_train, bert_model_name, max_len, train_batch_size)
    dr_validate, dl_validate = create_dataloader(df_validate, bert_model_name, max_len, validation_batch_size)

    # TODO: one training function, that contains everything below.    
    # Get inspiration from https://github.com/copenlu/stat-nlp-book/blob/master/labs/lab_3.ipynb
    optimizer_parameters = network.parameters()

    # Applying per-parameter weight-decay if chosen
    if custom_weight_decay:
        param_optimizer = list(network.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

    num_train_steps = int(len(dr_train) / train_batch_size * epochs)
    
    optimizer = AdamW(optimizer_parameters, lr = learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps = warmup_steps, num_training_steps = num_train_steps
    )

    losses = []
    best_loss = np.inf

    for epoch in range(epochs):
        
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        train_loss = train_(network, dl_train, optimizer, device, scheduler)
        losses.append(train_loss)
        valid_loss = validate_(network, dl_validate, device)

        print(f"Train Loss = {train_loss} Valid Loss = {valid_loss}")

        if valid_loss < best_loss:
            best_parameters = network.state_dict()            
            best_loss = valid_loss

    # return best model
    network.load_state_dict(best_parameters)

    return network, losses

if __name__ == '__main__':
    #df_train, df_validate = get_dane_data_split(['train', 'validate']) 
    # set to test_run_size to 0, if you want a complete training
    #test_run_size = 16
    #if test_run_size != 0:
    #    df_train = df_train[1:(test_run_size + 1)]
    #    df_validate = df_validate[1:(test_run_size + 1)]
    network, losses = train_model(epochs = 1,
                                  warmup_steps = 0,
                                  learning_rate = 0.0001,
                                  train_batch_size = 13)
    # plt.plot(losses)
    torch.save(network.state_dict(), "model.bin")
        
