
from NERDA.datasets import get_dane_data, download_dane_data
# TODO: should not be necesssary to download before importing NERDA.
# Download necessary ressources
download_dane_data()
from NERDA.models import NERDA
from NERDA.precooked import DA_ELECTRA_DA
import pandas as pd
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

#### TRAINING ####
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

#### PREDICTIONS ####

# set example tests.
text_single = "Pernille Rosenkrantz-Theil kommer fra Vejle"
# TODO: must work for a single sentence.
sentences = [nltk.word_tokenize(text_single)]

def test_predict():
    """Test that predict runs"""
    predictions = model.predict(sentences)


predictions = model.predict(sentences)

def test_predict_type():
    """Test token predictions"""
    assert isinstance(predictions, list)

def test_predict_length():
    """Test that sentence and prediction lenghts match"""
    assert len(sentences[0])==len(predictions[0])

def test_predict_text():
    """Test that predict_text runs"""
    predictions = model.predict_text(text_single)

def test_predict_maxlen_exceed():
    """That that exceeding max len does not break predict"""
    text = "ice " * 200
    sentences = [nltk.word_tokenize(text)]
    model.predict(sentences)

predictions_text_single = model.predict_text(text_single)

def test_predict_text_format():
    """Test text predictions"""
    assert isinstance(predictions_text_single, tuple)
    assert len(predictions_text_single[0][0]) == len(predictions_text_single[1][0])

# multiple sentences.
text_multi = """
Pernille Rosenkrantz-Theil kommer fra Vejle.
Jens Hansen har en bondegård.
"""

def test_predict_text_multi():
    """Test that predict_text runs with multiple sentences"""
    predictions = model.predict_text(text_multi, batch_size = 2)

predictions_text_multi = model.predict_text(text_multi, batch_size = 2)

def test_predict_text_multi_format():
    """Test multi-sentence text predictions"""
    assert isinstance(predictions_text_multi, tuple)
    assert len(predictions_text_multi[0]) == 2
    assert len(predictions_text_multi[1]) == 2
    assert len(predictions_text_multi[0][0]) == len(predictions_text_multi[1][0])
    assert len(predictions_text_multi[0][1]) == len(predictions_text_multi[1][1])

#### PERFORMANCE ####

def test_evaluate_performance():
    test = get_dane_data('test')
    f1 = model.evaluate_performance(test)
    assert isinstance(f1, pd.DataFrame)

#### PRECOOKED ####

from NERDA.precooked import DA_ELECTRA_DA

def test_load_precooked():
    """Test that precooked model can be (down)loaded, instantiated and works end-to-end"""
    m = DA_ELECTRA_DA()
    m.download_network()
    m.load_network()
    m.predict_text("Jens Hansen har en bondegård. Det har han!")













