from NERDA.models import NERDA
from NERDA.datasets import get_conll_data

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