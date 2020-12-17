from .train import train_model
from .data_generator import get_dane_data_split, encoder_
from .predict import predict, compute_performance
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

        self.hyperparameters = {'epochs' : 1,
                                'warmup_steps' : 0,
                                'train_batch_size': 5,
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
                              df = df,
                              max_len = 128,
                              device = self.device)

        return predictions

    def evaluate_performance(self, df):
        sentences, tags_predicted = self.predict(df)
        # TODO: move encoder_ to model.
        # encode tags.
        tags_actual = df['tags'].apply(encoder_.inverse_transform).tolist()
        performance = compute_performance(tags_predicted, tags_actual)
        return performance

if __name__ == '__main__':
    from NERDA.data_generator import get_dane_data_split
    from NERDA.models import NERDA
    N = NERDA(df_train = get_dane_data_split('train')[1:6],
              df_validate = get_dane_data_split('validate')[1:6])
    N.train()
    test = get_dane_data_split('test')[1:6]
    N.evaluate_performance(test)
    #torch.save(N.network.state_dict(), "model.bin")
    #N.load_network(model_path = "/home/ec2-user/NERDA/model.bin")
    # N.predict(rune_dane_inference = True, print_f1_scores = True)

    text = "Pernille Rosenkrantz-Theil kommer fra Vejle"
    import nltk
    words = nltk.word_tokenize(text)
    tags = [8] * len(words)
    import pandas as pd
    df = pd.DataFrame({'words': [words], 'tags': [tags]})
    sentences, predictions = N.predict(df = df)
    print(list(zip(sentences, predictions)))
    N.predict(df = df)
        
