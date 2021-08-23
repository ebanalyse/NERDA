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

test = get_dane_data('test')
perf = model.evaluate_performance(test)

def test_performance_df():
    assert isinstance(perf, pd.DataFrame)

def test_performance_len():
    assert len(perf) > 0

def test_includes_relevant_metrics():
    metrics = ['F1-Score', 'Precision', 'Recall']
    assert all([x in perf.columns for x in metrics])

def test_metrics_dtype():
    metrics = ['F1-Score', 'Precision', 'Recall']
    assert all([perf.dtypes[x] == 'float' for x in metrics])

