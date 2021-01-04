from NERDA.models import NERDA
from NERDA.datasets import get_dane_data, download_dane_data
import nltk
nltk.download('punkt')
download_dane_data()

# instantiate model.
model = NERDA(dataset_training = get_dane_data('train', 5),
              dataset_validation = get_dane_data('validate', 5),
              transformer = 'bert-base-multilingual-uncased')

def test_instantiate_NERDA():
    assert isinstance(model, NERDA)

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

