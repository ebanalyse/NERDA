from NERDA.models import NERDA
from NERDA.datasets import get_dane_data
import pickle
import torch
import boto3

def deploy_model_to_s3(model, test_set = get_dane_data('test')):

    model_name = type(model).__name__

    file_model = f'{model_name}.bin'
    torch.save(model.network.state_dict(), file_model)
    
    # compute performance on test set and save.
    performance = model.evaluate_performance(test_set)
    # TODO: save as .csv
    file_performance = f'{model_name}_performance.pickle'
    with open(file_performance, 'wb') as file:
        pickle.dump(file_performance, file)
    
    # upload to S3 bucket.
    s3 = boto3.resource('s3')
    s3.Bucket('nerda').upload_file(
            Filename=file_model, 
            Key = file_model)
    s3.Bucket('nerda').upload_file(
            Filename=file_performance, 
            Key = file_performance)

    return "Model deployed to S3 successfully."   

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
