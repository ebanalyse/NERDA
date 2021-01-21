from NERDA.datasets import get_dane_data
from NERDA.models import NERDA
import nltk

# instantiate a minimal model.
model = NERDA(dataset_training = get_dane_data('train', 5),
              dataset_validation = get_dane_data('dev', 5),
              transformer = 'Maltehb/-l-ctra-danish-electra-small-uncased',
              hyperparameters = {'epochs' : 1,
                                 'warmup_steps' : 10,
                                 'train_batch_size': 5,
                                 'learning_rate': 0.0001})


# set example texts to identify entities in.
text_single = "Pernille Rosenkrantz-Theil kommer fra Vejle"
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

def test_predict_text_match_words_predictions():
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
    """Test multi-sentence text predictions has expected format"""
    assert isinstance(predictions_text_multi, tuple)

def test_predict_text_multi_elements_count():
    """Test dimensions of multi-sentence text predictions"""
    assert [len(predictions_text_multi[0]), len(predictions_text_multi[1])] == [2, 2]

def test_predict_text_multi_lens():
    """Test lengths of multi-sentence text predictions"""
    s1 = len(predictions_text_multi[0][0]) == len(predictions_text_multi[1][0])
    s2 = len(predictions_text_multi[0][1]) == len(predictions_text_multi[1][1])
    assert all([s1, s2])

