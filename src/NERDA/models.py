"""NERDA models"""
from .datasets import get_dane_data
from .networks import NERDANetwork
from .predictions import predict, predict_text
from .performance import compute_f1_scores
from .training import train_model
import pandas as pd
import torch
import os
from typing import List
from sklearn import preprocessing
from transformers import AutoModel, AutoTokenizer, AutoConfig

class NERDA:
    """NERDA model

    A NERDA model object containing a complete model configuration.
    The model can be trained with the 'train' method. Afterwards
    new observations can be predicted with 'predict*' methods.

    Examples:
        Model for a VERY small subset of Danish NER data
        >>> from NERDA.dataset import get_dane_data
        >>> trn = get_dane_data('train', 5)
        >>> valid = get_dane_data('dev', 5)
        >>> tag_scheme = ['B-PER', 'I-PER' 'B-LOC', 'I-LOC',
                          'B-ORG', 'I-ORG', 'B-MISC, 'I-MISC']
        >>> tag_outside = 'O'
        >>> transformer = 'bert-base-multilingual-uncased',
        >>> model = NERDA(transformer = transformer,
                          tag_scheme = tag_scheme,
                          tag_outside = tag_outside,
                          dataset_training = trn,
                          dataset_validation = valid)

        Model for complete Danish NER data (DaNE) with modified hyperparameters
        >>> trn = get_dane_data('train')
        >>> valid = get_dane_data('dev')
        >>> transformer = 'bert-base-multilingual-uncased',
        >>> hyperparameters = {'epochs' : 3,
                               'warmup_steps' : 400,
                               'train_batch_size': 16,
                               'learning_rate': 0.0001},
        >>> model = NERDA(transformer = transformer,
                          dataset_training = trn,
                          dataset_validation = valid,
                          tag_scheme = tag_scheme,
                          tag_outside = tag_outside,
                          dropout = 0.1,
                          hyperparameters = hyperparameters)

    Attributes:
        network (NERDANetwork): network for Named Entity 
            Recognition task.
        tag_encoder (preprocessing.LabelEncoder): encoder for the
            NER labels/tags.
        transformer_model (AutoModel): (Auto)Model derived from the
            transformer.
        transformer_tokenizer (AutoTokenizer): (Auto)Tokenizer
            derived from the transformer.
        transformer_config (AutoConfig): (Auto)Config derived from
            the transformer. 
        losses (list): holds training losses, when the model has been 
            trained.
    """
    def __init__(self, 
                 transformer: str = 'bert-base-multilingual-uncased',
                 device: str = None, 
                 tag_scheme: List[str] = [
                            'B-PER',
                            'I-PER', 
                            'B-ORG', 
                            'I-ORG', 
                            'B-LOC', 
                            'I-LOC', 
                            'B-MISC', 
                            'I-MISC'
                            ],
                 tag_outside: str = 'O',
                 dataset_training: dict = None,
                 dataset_validation: dict = None,
                 max_len: int = 130,
                 dropout: float = 0.1,
                 hyperparameters: dict = {'epochs' : 3,
                                          'warmup_steps' : 500,
                                          'train_batch_size': 16,
                                          'learning_rate': 0.0001},
                 tokenizer_parameters: dict = {'do_lower_case' : True},
                 validation_batch_size: int = 8,
                 num_workers: int = 1) -> None:
        """Initialize NERDA model

        Args:
            transformer (str, optional): which pretrained 'huggingface' 
                transformer to use. 
            device (str, optional): the desired device to use for computation. 
                If not provided by the user, we take a guess.
            tag_scheme (List[str], optional): [description]. All available NER 
                tags for the given data set EXCLUDING the special outside tag, 
                that is handled separately.
            tag_outside (str, optional): the value of the special outside tag. 
                Defaults to 'O'.
            dataset_training (dict, optional): the training data. Must consist 
                of 'sentences': word-tokenized sentences and 'tags': corresponding 
                NER tags. Defaults to None, in which case the DaNE data set is used.
            dataset_validation (dict, optional): the validation data. Must consist
                of 'sentences': word-tokenized sentences and 'tags': corresponding 
                NER tags. Defaults to None, in which case the DaNE data set is used.
            max_len (int, optional): the maximum sentence length (number of 
                tokens after applying the transformer tokenizer) for the transformer. 
                Sentences are truncated accordingly.
            dropout (float, optional): dropout probability. Defaults to 0.1.
            hyperparameters (dict, optional): Hyperparameters for the model. Defaults
                to {'epochs' : 3, 'warmup_steps' : 500, 'train_batch_size': 16, 
                'learning_rate': 0.0001}.
            tokenizer_parameters (dict, optional): parameters for the transformer 
                tokenizer. Defaults to {'do_lower_case' : True}.
            validation_batch_size (int, optional): batch size for validation. Defaults
                to 8.
            num_workers (int, optional): number of workers for data loader.
        """
        
        # set device automatically if not provided by user.
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print("Device automatically set to:", self.device)
        self.tag_scheme = tag_scheme
        self.tag_outside = tag_outside
        self.transformer = transformer
        self.max_len = max_len
        if dataset_training is None:
            dataset_training = get_dane_data('train')
        if dataset_validation is None:
            dataset_validation = get_dane_data('dev')     
        self.dataset_training = dataset_training
        self.dataset_validation = dataset_validation
        self.hyperparameters = hyperparameters
        self.tag_outside = tag_outside
        self.tag_scheme = tag_scheme
        tag_complete = [tag_outside] + tag_scheme
        # fit encoder to _all_ possible tags.
        self.tag_encoder = preprocessing.LabelEncoder()
        self.tag_encoder.fit(tag_complete)
        self.transformer_model = AutoModel.from_pretrained(transformer)
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(transformer, **tokenizer_parameters)
        self.transformer_config = AutoConfig.from_pretrained(transformer)  
        self.network = NERDANetwork(self.transformer_model, self.device, len(tag_complete), dropout = dropout)
        self.network.to(self.device)
        self.validation_batch_size = validation_batch_size
        self.num_workers = num_workers
        self.losses = []

    def train(self) -> str:
        """Train Network

        Trains the network from the NERDA model specification.

        Returns:
            str: a message saying if the model was trained succesfully.
            The network in the 'network' attribute is trained as a 
            side-effect. Training losses are also saved in 'losses' 
            attribute as a side-effect.
        """
        network, losses = train_model(network = self.network,
                                      tag_encoder = self.tag_encoder,
                                      tag_outside = self.tag_outside,
                                      transformer_tokenizer = self.transformer_tokenizer,
                                      transformer_config = self.transformer_config,
                                      dataset_training = self.dataset_training,
                                      dataset_validation = self.dataset_validation,
                                      validation_batch_size = self.validation_batch_size,
                                      max_len = self.max_len,
                                      device = self.device,
                                      num_workers = self.num_workers,
                                      **self.hyperparameters)
        
        # attach as attributes to class
        setattr(self, "network", network)
        setattr(self, "losses", losses)

        return "Model trained successfully"

    def load_network_from_file(self, model_path = "model.bin") -> str:
        """Load Pretrained NERDA Network from file

        Loads weights for a pretrained NERDA Network from file.

        Args:
            model_path (str, optional): Path for model file. 
                Defaults to "model.bin".

        Returns:
            str: message telling if weights for network were
            loaded succesfully.
        """
        # TODO: change assert to Raise.
        assert os.path.exists(model_path), "File does not exist. You can download network with download_network()"
        self.network.load_state_dict(torch.load(model_path))
        return f'Weights for network loaded from {model_path}'

    def predict(self, sentences: List[List[str]], **kwargs) -> List[List[str]]:
        """Predict Named Entities in Word-Tokenized Sentences

        Predicts word-tokenized sentences with trained model.

        Args:
            sentences (List[List[str]]): word-tokenized sentences.
            kwargs: arbitrary keyword arguments. For instance
                'batch_size' and 'num_workers'.

        Returns:
            List[List[str]]: Predicted tags for sentences - one
            predicted tag/entity per word token.
        """
        return predict(network = self.network, 
                       sentences = sentences,
                       transformer_tokenizer = self.transformer_tokenizer,
                       transformer_config = self.transformer_config,
                       max_len = self.max_len,
                       device = self.device,
                       tag_encoder = self.tag_encoder,
                       tag_outside = self.tag_outside,
                       **kwargs)

    def predict_text(self, text: str, **kwargs) -> list:
        """Predict Named Entities in a Text

        Args:
            text (str): text to predict entities in.
            kwargs: arbitrary keyword arguments. For instance
                'batch_size' and 'num_workers'.

        Returns:
            List with word-tokenized sentences and predicted 
            tags/entities.
        """
        return predict_text(network = self.network, 
                            text = text,
                            transformer_tokenizer = self.transformer_tokenizer,
                            transformer_config = self.transformer_config,
                            max_len = self.max_len,
                            device = self.device,
                            tag_encoder = self.tag_encoder,
                            tag_outside = self.tag_outside,
                            **kwargs)

    def evaluate_performance(self, dataset: dict) -> pd.DataFrame:
        """Evaluate Performance

        Evaluates the performance of the model on an arbitrary
        data set.

        Args:
            dataset (dict): Data set that must consist of
                'sentences' and NER'tags'.

        Returns:
            DataFrame with performance numbers, F1-scores.
        """
        
        tags_predicted = self.predict(dataset.get('sentences'))
        
        f1 = compute_f1_scores(y_pred = tags_predicted, 
                               y_true = dataset.get('tags'),
                               labels = self.tag_scheme,
                               average = None)
        
        # create DataFrame with performance scores (=F1)
        df = list(zip(self.tag_scheme, f1[2]))
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