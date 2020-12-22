from .training import train_model
from .datasets import get_dane_data
from .predictions import predict
from .performance import compute_f1_scores
from .networks import GenericNetwork
import pandas as pd
from sklearn import preprocessing
from transformers import AutoModel, AutoTokenizer
import torch

class NERDA():

    def __init__(self, 
                transformer = 'bert-base-multilingual-uncased',
                device = None, 
                tag_scheme = [
                            'B-PER',
                            'I-PER', 
                            'I-ORG', 
                            'B-ORG', 
                            'B-LOC', 
                            'I-LOC', 
                            'B-MISC', 
                            'I-MISC'
                            ],
                tag_outside = 'O',
                dataset_training = get_dane_data('train'),
                dataset_validation = get_dane_data('validate'),
                do_lower_case = True,
                max_len = 128,
                dropout = 0.1):
        
        # set device automatically if not provided by user.
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print("Device automatically set to:", self.device)
        self.tag_scheme = tag_scheme
        self.tag_outside = tag_outside
        self.transformer = transformer
        self.max_len = max_len
        self.dataset_training = dataset_training
        self.dataset_validation = dataset_validation
        self.hyperparameters = {'epochs' : 1,
                                'warmup_steps' : 0,
                                'train_batch_size': 5,
                                'learning_rate': 0.0001}
        self.tag_outside = tag_outside
        self.tag_scheme = tag_scheme
        tag_complete = [tag_outside] + tag_scheme
        # fit encoder to _all_ possible tags.
        self.tag_encoder = preprocessing.LabelEncoder()
        self.tag_encoder.fit(tag_complete)
        self.transformer_model = AutoModel.from_pretrained(transformer)
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(transformer, do_lower_case = do_lower_case)  
        self.network = GenericNetwork(self.transformer_model, self.device, len(tag_complete), dropout = dropout)
        self.network.to(self.device)

    def train(self):
        network, losses = train_model(network = self.network,
                                      tag_encoder = self.tag_encoder,
                                      transformer_tokenizer = self.transformer_tokenizer,
                                      dataset_training = self.dataset_training,
                                      dataset_validation = self.dataset_validation,
                                      max_len = self.max_len,
                                      device = self.device,
                                      **self.hyperparameters)
        
        # attach as attributes to class
        setattr(self, "network", network)
        setattr(self, "losses", losses)

        return network, losses

    def load_network(self, model_path = "model.bin"):
        self.network.load_state_dict(torch.load(model_path))
        return f'Weights for network loaded from {model_path}'

    def predict(self, sentences):
        predictions = predict(network = self.network, 
                              sentences = sentences,
                              transformer_tokenizer = self.transformer_tokenizer,
                              max_len = self.max_len,
                              device = self.device,
                              tag_encoder = self.tag_encoder)

        return predictions

    def evaluate_performance(self, dataset):
        
        tags_predicted = self.predict(dataset.get('sentences'))
        
        f1 = compute_f1_scores(y_pred = tags_predicted, 
                               y_true = dataset.get('tags'),
                               labels = self.tag_scheme,
                               average = None)
        
        # create DataFrame with performance scores (=F1)
        df = list(zip(self.tag_scheme, f1[2]))
        # TODO: overvej om pandas skal udgå
        df = pd.DataFrame(df, columns = ['Level', 'F1-Score'])    
        
        # compute MICRO-averaged F1-scores and add to table.
        f1_micro = compute_f1_scores(y_pred = tags_predicted, 
                                     y_true = dataset.get('tags'),
                                     labels = self.tag_scheme,
                                     average = 'micro')
        f1_micro = pd.DataFrame({'Level' : ['AVG_MICRO'], 'F1-Score': [f1_micro[2]]})
        df = df.append(f1_micro)

        # compute MACRO-averaged F1-scores and add to table.
        f1_macro = compute_f1_scores(y_pred = tags_predicted, 
                                     y_true = dataset.get('tags'),
                                     labels = self.tag_scheme,
                                     average = 'macro')
        f1_macro = pd.DataFrame({'Level' : ['AVG_MACRO'], 'F1-Score': [f1_macro[2]]})
        df = df.append(f1_macro)
     
        return df

if __name__ == '__main__':
    from NERDA.datasets import get_dane_data
    from NERDA.models import NERDA
    t = 'bert-base-multilingual-uncased'
    # t = 'Maltehb/-l-ctra-danish-electra-small-uncased'
    # t = 'xlm-roberta-base' # predicter I-MISC
    # t = 'distilbert-base-multilingual-cased' # TODO: forward tager ikke 'token_type_ids', fejler -> Fjern? 
    N = NERDA(dataset_training = get_dane_data('train', 5),
              dataset_validation = get_dane_data('validate', 5),
              transformer = t)
    N.train()
    dataset_test = get_dane_data('test', 5)
    f1 = N.evaluate_performance(dataset_test)
    #torch.save(N.network.state_dict(), "model.bin")
    #N.load_network(model_path = "/home/ec2-user/NERDA/model.bin")

    text = "Pernille Rosenkrantz-Theil kommer fra Vejle"
    import nltk
    # TODO: must work for a single sentence.
    sentences = [nltk.word_tokenize(text)]
    predictions = N.predict(sentences)
    print(list(zip(sentences, predictions)))

    #m = AutoModel.from_pretrained('bert-base-multilingual-uncased')
    #config = AutoConfig.from_pretrained('Maltehb/-l-ctra-danish-electra-small-uncased')
    #m = AutoConfig.from_pretrained('bert-base-multilingual-uncased')
    

        
