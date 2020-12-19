from NERDA.models import NERDA
from NERDA.datasets import get_dane_data
import nltk

# instantiate model.
model = NERDA(df_train = get_dane_data('train')[1:6],
              df_validate = get_dane_data('validate')[1:6])

def test_instantiate_NERDA():
    assert isinstance(model, NERDA)

m = model.train()

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

