from .training import train_model
from .datasets import get_dane_data
from .predictions import predict, compute_performance
from .networks import NER_BERT
from sklearn import preprocessing
from transformers import BertModel, BertTokenizer
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
                max_len = 128):
        
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
        # TODO: hmm, maybe independent of BertModel
        self.transformer_model = BertModel.from_pretrained(transformer)
        self.transformer_tokenizer = BertTokenizer.from_pretrained(transformer, do_lower_case = True)  
        self.network = NER_BERT(self.transformer_model, self.device, len(tag_complete))
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
        performance = compute_performance(tags_predicted, 
                                          dataset.get('tags'),
                                          tag_scheme = self.tag_scheme)
        return performance

if __name__ == '__main__':
    from NERDA.datasets import get_dane_data
    from NERDA.models import NERDA
    N = NERDA(dataset_training = get_dane_data('train', 5),
              dataset_validation = get_dane_data('validate', 5))
    N.train()
    dataset_test = get_dane_data('test', 5)
    N.evaluate_performance(dataset_test)
    #torch.save(N.network.state_dict(), "model.bin")
    #N.load_network(model_path = "/home/ec2-user/NERDA/model.bin")

    text = "Pernille Rosenkrantz-Theil kommer fra Vejle"
    import nltk
    # TODO: must work for a single sentence.
    sentences = [nltk.word_tokenize(text)]
    predictions = N.predict(sentences)
    print(list(zip(sentences, predictions)))

        
