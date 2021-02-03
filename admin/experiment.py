import pandas as pd
from functools import reduce
from NERDA.datasets import get_conll_data
from NERDA.models import NERDA

def train_n(n = 5):
    
    print(f'Training with {n} observations')
    model = NERDA(transformer = 'google/electra-small-discriminator',
                  tag_scheme = [
                      'B-PER',
                      'I-PER', 
                      'B-ORG', 
                      'I-ORG', 
                      'B-LOC', 
                      'I-LOC', 
                      'B-MISC', 
                      'I-MISC'
                      ],
                  tag_outside = 'O',
                  dataset_training = get_conll_data('train', n),
                  dataset_validation = get_conll_data('valid'),
                  max_len = 128,
                  dropout = 0.1,
                  hyperparameters = {'epochs' : 4,
                                      'warmup_steps' : 250,
                                      'train_batch_size': 13,
                                      'learning_rate': 8e-05},
                  tokenizer_parameters = {'do_lower_case' : True})

    model.train()

    performance = model.evaluate_performance(get_conll_data('test'))

    performance = performance.append({'Level': 'N_SENTS', 'F1-Score' : n}, ignore_index=True)

    performance = performance.rename(columns = {"F1-Score" : f'N{n}'})

    print(performance)

    return performance

#training_sizes = [5, 10]
#out = [train_n(n) for n in training_sizes]

out = [train_n(n) for n in range(1039, len(get_conll_data('train')['sentences']) + 1000, 1000)]

out = reduce(lambda x, y: pd.merge(x, y), out)

out.to_csv("performance_versus_training_size.csv", index = False, sep = ";")
