from .models import NERDA
from .datasets import get_dane_data
import pickle
import torch
import boto3

def train_save(transformer = 'bert-base-multilingual-uncased',
               alias = 'mbert',
               hyperparameters = {'epochs' : 4,
                                  'warmup_steps' : 500,
                                  'train_batch_size': 13,
                                  'learning_rate': 0.0001},
               limit = None,
               to_s3 = False):
    
    # instantiate model.
    model = NERDA(transformer = transformer,
                  dataset_training = get_dane_data('train', limit = limit), 
                  dataset_validation = get_dane_data('validate', limit = limit),
                  hyperparameters = {'epochs' : 4,
                                     'warmup_steps' : 500,
                                     'train_batch_size': 13,
                                     'learning_rate': 0.0001})
    model.train()

    # save model.
    file_model = f'{alias}_model.bin'
    torch.save(model.network.state_dict(), file_model)
    
    # compute performance on test set and save.
    dataset_test = get_dane_data('test', limit = limit)
    f1 = model.evaluate_performance(dataset_test)
    file_f1 = f'{alias}_f1.pickle'
    with open(file_f1, 'wb') as file:
        pickle.dump(f1, file)
    
    # save in S3 bucket.
    if to_s3:
        s3 = boto3.resource('s3')
        s3.Bucket('larsktest').upload_file(
            Filename=file_model, 
            Key = f'NERDA/{file_model}')
        s3.Bucket('larsktest').upload_file(
            Filename=file_f1, 
            Key = f'NERDA/{file_f1}')

    return model, f1   

cfg_mbert = {
    'transformer': 'bert-base-multilingual-uncased',
    'alias': 'mbert',
    'hyperparameters': {'epochs' : 4,
                        'warmup_steps' : 500,
                        'train_batch_size': 13,
                        'learning_rate': 0.0001}
    }

cfg_xlmroberta = {
    'transformer': 'xlm-roberta-base',
    'alias': 'xlmroberta',
    'hyperparameters': {'epochs' : 3,
                        'warmup_steps' : 500,
                        'train_batch_size': 13,
                        'learning_rate': 0.0001}
    }

cfg_dabert = {
    'transformer': 'DJSammy/bert-base-danish-uncased_BotXO,ai',
    'alias': 'dabert',
    'hyperparameters': {'epochs' : 4,
                        'warmup_steps' : 500,
                        'train_batch_size': 13,
                        'learning_rate': 0.0001}
    }

cfg_electra = {
    'transformer': 'Maltehb/-l-ctra-danish-electra-small-uncased',
    'alias': 'electra',
    'hyperparameters': {'epochs' : 4,
                        'warmup_steps' : 500,
                        'train_batch_size': 13,
                        'learning_rate': 0.0001}
    }

cfg_electra = {
    'transformer': 'distilbert-base-multilingual-cased',
    'alias': 'distilbert',
    'hyperparameters': {'epochs' : 4,
                        'warmup_steps' : 500,
                        'train_batch_size': 13,
                        'learning_rate': 0.0001}
    }

if __name__ == '__main__':
    from NERDA.train_save import train_save, cfg_electra
    m, f1 = train_save(cfg_electra)
