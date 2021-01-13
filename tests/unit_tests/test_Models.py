from NERDA.datasets import get_dane_data, download_dane_data
# TODO: should not be necesssary to download before importing NERDA.

# Download necessary ressources
download_dane_data()
from NERDA.models import NERDA
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

def test_training():
    """Test if training runs successfully"""
    model.train()

text = "Pernille Rosenkrantz-Theil kommer fra Vejle"
import nltk
# TODO: must work for a single sentence.
sentences = [nltk.word_tokenize(text)]
predictions = model.predict(sentences)

def test_predict():
    assert isinstance(predictions, list)

def test_predict_length():
    assert len(sentences[0])==len(predictions[0])

#text = "Pernille Rosenkrantz-Theil kommer fra Vejle"
#words = nltk.word_tokenize(text)
#tags = [8] * len(words)
#df = pd.DataFrame({'words': [words], 'tags': [tags]})
#sentences, predictions = model.predict(df = df)
#
#def test_predictions():
#    assert isinstance(predictions, list)
#    assert len(sentences) == len(predictions)
#    len_sens = [len(sentence) for sentence in sentences]
#    len_preds = [len(prediction) for prediction in predictions]
#    assert len_sens == len_preds
#
#test = get_dane_data_split('test')[1:6]
#performance = model.evaluate_performance(test)
#
#def test_evaluate_performance():
#    assert isinstance(performance, list)

