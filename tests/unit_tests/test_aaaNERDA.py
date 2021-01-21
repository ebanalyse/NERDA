# HACK: Filename prefixed with 'aaa' to execute this test before the others
# in order to download necessary ressources for all other tests.

from NERDA.datasets import get_dane_data, download_dane_data
# TODO: should not be necesssary to download before importing NERDA.
# Download necessary ressources
download_dane_data()
from NERDA.models import NERDA
from NERDA.precooked import DA_ELECTRA_DA
import nltk
nltk.download('punkt')

# instantiate a minimal model.
model = NERDA(dataset_training = get_dane_data('train', 5),
              dataset_validation = get_dane_data('dev', 5),
              max_len = 128,
              transformer = 'Maltehb/-l-ctra-danish-electra-small-uncased',
              hyperparameters = {'epochs' : 1,
                                 'warmup_steps' : 10,
                                 'train_batch_size': 5,
                                 'learning_rate': 0.0001})

def test_instantiate_NERDA():
    """Test that model has the correct/expected class"""
    assert isinstance(model, NERDA)















