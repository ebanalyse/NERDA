from NERDA.datasets import get_dane_data
from NERDA.models import NERDA
import pandas as pd

# instantiate a minimal model.
model = NERDA(dataset_training = get_dane_data('train', 5),
              dataset_validation = get_dane_data('dev', 5),
              transformer = 'Maltehb/-l-ctra-danish-electra-small-uncased',
              hyperparameters = {'epochs' : 1,
                                 'warmup_steps' : 10,
                                 'train_batch_size': 5,
                                 'learning_rate': 0.0001})

def test_evaluate_performance():
    test = get_dane_data('test')
    f1 = model.evaluate_performance(test)
    assert isinstance(f1, pd.DataFrame)
    assert len(f1) > 0