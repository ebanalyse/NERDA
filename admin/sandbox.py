from NERDA.models import NERDA
from NERDA.datasets import get_conll_data, get_dane_data
from transformers import AutoTokenizer
trans = 'bert-base-multilingual-uncased'
tokenizer = AutoTokenizer.from_pretrained(trans, do_lower_case = True)
data = get_dane_data('train')

sents = data.get('sentences')

out = []

for sent in sents:
    sent = sents[3595]
    tokens = []
    for word in sent:
            tokens.extend(tokenizer.tokenize(word))
    out.append(tokens)

lens = [len(x) for x in out]

max(lens)

sents[3595]


from transformers import AutoTokenizer, AutoModel, AutoConfig 
t = 'google/electra-small-discriminator'
cfg = AutoModel.from_pretrained(t)











#trn = get_conll_data('train')
#idx_min = 3110
#idx_max = 3115
#valid = get_conll_data('valid')
#valid['sentences'] = valid['sentences'][idx_min:idx_max+1]
#valid['tags'] = valid['tags'][idx_min:idx_max+1]
#trn['sentences'] = trn['sentences'][idx_min:idx_max+1]
#trn['tags'] = trn['tags'][idx_min:idx_max+1]
# model = NERDA(dataset_training=trn,
 #             dataset_validation = valid)
#model.train()
#k=0
#trn['sentences'][3111]
#from transformers import AutoTokenizer
#t = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
#valid = get_conll_data('valid')

filename = 'en_bert_ml.pkl'
# pickle.dump(model, open(filename, 'wb'))
import pickle
file = open(filename,'rb')
model = pickle.load(file) 
test = get_conll_data('test')
model.evaluate_performance(test, batch_size = 10)
#for entry in range(3120,3140):
#    print(entry)
#    sent = trn['sentences'][entry]
#    [t.tokenize(word) for word in sent]

test = get_conll_data('test')
idx_min = 202
idx_max = 202
# valid = get_conll_data('valid')
#valid['sentences'] = valid['sentences'][idx_min:idx_max+1]
#valid['tags'] = valid['tags'][idx_min:idx_max+1]
test['sentences'] = test['sentences'][idx_min:idx_max+1]
test['tags'] = test['tags'][idx_min:idx_max+1]
model.evaluate_performance(test)
# model = NERDA(dataset_training=trn,
 #             dataset_validation = valid)
#model.train()
#k=0
#trn['sentences'][3111]
#from transformers import AutoTokenizer
#t = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
#valid = get_conll_data('valid')

<<<<<<< HEAD:admin/sandbox.py

transformer = "google/electra-small-discriminator"
from transformers import AutoTokenizer, AutoModel, AutoConfig 
trans = AutoConfig.from_pretrained(transformer)

def tester():

    try:
        model = AutoModel.from_pretrained('google/electra-small-discriminator')
    except:
        print("Oops!", sys.exc_info()[0], "occurred.")

    return model
=======
from NERDA.datasets import get_dane_data
trn = get_conll_data('train', 5)
valid = get_conll_data('dev', 5)
transformer = 'bert-base-multilingual-uncased',
model = NERDA(transformer = transformer,
              dataset_training = trn,
              dataset_validation = valid)
>>>>>>> b5eea087ece5f61ec70aa3f99cd4c99b418ebb92:sandbox.py
