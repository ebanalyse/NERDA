from .train import train_model
from .data_generator import get_dane_data_split
from .predict import predict
import torch

from .model import NER_BERT

class NERDA():

    def __init__(self, 
                transformer = 'bert-base-multilingual-uncased',
                device = None, 
                tag_scheme = ['O', 
                            'B-PER',
                            'I-PER', 
                            'I-ORG', 
                            'B-ORG', 
                            'B-LOC', 
                            'I-LOC', 
                            'B-MISC', 
                            'I-MISC'],
                df_train = get_dane_data_split('train'),
                df_validate = get_dane_data_split('validate')):
        
        # set device automatically if not provided by user.
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print("Device automatically set to:", self.device)

        self.tag_scheme = tag_scheme
        self.transformer = transformer
        
        self.network = NER_BERT(transformer, self.device, len(tag_scheme))
        self.network.to(self.device)
        self.df_train = df_train
        self.df_validate = df_validate

        self.hyperparameters = {'epochs' : 4,
                                'warmup_steps' : 500,
                                'train_batch_size': 13,
                                'learning_rate': 0.0001}

    def train(self):
        network, losses = train_model(network = self.network,
                                      bert_model_name = self.transformer,
                                      df_train = self.df_train,
                                      df_validate = self.df_validate,
                                      device = self.device,
                                      **self.hyperparameters)
        
        # attach as attributes to class
        setattr(self, "network", network)
        setattr(self, "losses", losses)

        return network, losses

    def load_network(self, model_path = "model.bin"):
        self.network.load_state_dict(torch.load(model_path))
        return f'Weights for network loaded from {model_path}'

    def predict(self, df):
        predictions = predict(network = self.network, 
                              df_test = df,
                              run_dane_inference = False,
                              print_f1_scores = False,
                              max_len = 128,
                              device = self.device,
                              model_path = "model.bin")

        return predictions

if __name__ == '__main__':
    N = NERDA()
    #N.train()
    #torch.save(N.network.state_dict(), "model.bin")
    # N.load_network()

    text = "Pernille Rosenkrantz-Theil kommer fra Vejle"
    import nltk
    words = nltk.word_tokenize(text)
    tags = [8] * len(words)
    import pandas as pd
    df = pd.DataFrame({'words': [words], 'tags': [tags]})
    sentences, predictions = N.predict(df = df)
    print(list(zip(sentences, predictions)))
        
