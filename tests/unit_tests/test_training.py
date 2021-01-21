from NERDA.datasets import get_dane_data
from NERDA.models import NERDA

# instantiate a minimal model.
model = NERDA(dataset_training = get_dane_data('train', 5),
              dataset_validation = get_dane_data('dev', 5),
              transformer = 'Maltehb/-l-ctra-danish-electra-small-uncased',
              hyperparameters = {'epochs' : 1,
                                 'warmup_steps' : 10,
                                 'train_batch_size': 5,
                                 'learning_rate': 0.0001})

def test_training():
    """Test if training runs successfully"""
    model.train()

def test_training_exceed_maxlen():
    """Test if traning does not break even though MAX LEN is exceeded"""
    m = NERDA(dataset_training = get_dane_data('train', 5),
              dataset_validation = get_dane_data('dev', 5),
              max_len = 3,
              transformer = 'Maltehb/-l-ctra-danish-electra-small-uncased',
              hyperparameters = {'epochs' : 1,
                                 'warmup_steps' : 10,
                                 'train_batch_size': 5,
                                 'learning_rate': 0.0001})
    m.train()

def test_training_bert():
    """Test if traning does not break even though MAX LEN is exceeded"""
    m = NERDA(dataset_training = get_dane_data('train', 5),
              dataset_validation = get_dane_data('dev', 5),
              transformer = 'bert-base-multilingual-uncased',
              hyperparameters = {'epochs' : 1,
                                 'warmup_steps' : 10,
                                 'train_batch_size': 5,
                                 'learning_rate': 0.0001})
    m.train()
