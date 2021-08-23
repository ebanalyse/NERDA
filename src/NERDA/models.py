"""
This section covers the interface for `NERDA` models, that is 
implemented as its own Python class [NERDA.models.NERDA][].

The interface enables you to easily 

- specify your own [NERDA.models.NERDA][] model
- train it
- evaluate it
- use it to predict entities in new texts.
"""
from NERDA.datasets import get_conll_data
from NERDA.networks import NERDANetwork
from NERDA.predictions import predict, predict_text
from NERDA.performance import compute_f1_scores, flatten
from NERDA.training import train_model
import pandas as pd
import numpy as np
import torch
import os
import sys
import sklearn.preprocessing
from sklearn.metrics import accuracy_score
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import List

class NERDA:
    """NERDA model

    A NERDA model object containing a complete model configuration.
    The model can be trained with the `train` method. Afterwards
    new observations can be predicted with the `predict` and
    `predict_text` methods. The performance of the model can be
    evaluated on a set of new observations with the 
    `evaluate_performance` method.

    Examples:
        Model for a VERY small subset (5 observations) of English NER data
        >>> from NERDA.datasets import get_conll_data
        >>> trn = get_conll_data('train', 5)
        >>> valid = get_conll_data('valid', 5)
        >>> tag_scheme = ['B-PER', 'I-PER', 'B-LOC', 'I-LOC',
                          'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC']
        >>> tag_outside = 'O'
        >>> transformer = 'bert-base-multilingual-uncased'
        >>> model = NERDA(transformer = transformer,
                          tag_scheme = tag_scheme,
                          tag_outside = tag_outside,
                          dataset_training = trn,
                          dataset_validation = valid)

        Model for complete English NER data set CoNLL-2003 with modified hyperparameters
        >>> trn = get_conll_data('train')
        >>> valid = get_conll_data('valid')
        >>> transformer = 'bert-base-multilingual-uncased'
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
        network (torch.nn.Module): network for Named Entity 
            Recognition task.
        tag_encoder (sklearn.preprocessing.LabelEncoder): encoder for the
            NER labels/tags.
        transformer_model (transformers.PreTrainedModel): (Auto)Model derived from the
            transformer.
        transformer_tokenizer (transformers.PretrainedTokenizer): (Auto)Tokenizer
            derived from the transformer.
        transformer_config (transformers.PretrainedConfig): (Auto)Config derived from
            the transformer. 
        train_losses (list): holds training losses, once the model has been 
            trained.
        valid_loss (float): holds validation loss, once the model has been trained.
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
                 max_len: int = 128,
                 network: torch.nn.Module = NERDANetwork,
                 dropout: float = 0.1,
                 hyperparameters: dict = {'epochs' : 4,
                                          'warmup_steps' : 500,
                                          'train_batch_size': 13,
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
            tag_scheme (List[str], optional): All available NER 
                tags for the given data set EXCLUDING the special outside tag, 
                that is handled separately.
            tag_outside (str, optional): the value of the special outside tag. 
                Defaults to 'O'.
            dataset_training (dict, optional): the training data. Must consist 
                of 'sentences': word-tokenized sentences and 'tags': corresponding 
                NER tags. You can look at examples of, how the dataset should 
                look like by invoking functions get_dane_data() or get_conll_data().
                Defaults to None, in which case the English CoNLL-2003 data set is used. 
            dataset_validation (dict, optional): the validation data. Must consist
                of 'sentences': word-tokenized sentences and 'tags': corresponding 
                NER tags. You can look at examples of, how the dataset should 
                look like by invoking functions get_dane_data() or get_conll_data().
                Defaults to None, in which case the English CoNLL-2003 data set 
                is used.
            max_len (int, optional): the maximum sentence length (number of 
                tokens after applying the transformer tokenizer) for the transformer. 
                Sentences are truncated accordingly. Look at your data to get an 
                impression of, what could be a meaningful setting. Also be aware 
                that many transformers have a maximum accepted length. Defaults 
                to 128. 
            network (torch.nn.module, optional): network to be trained. Defaults
                to a default generic `NERDANetwork`. Can be replaced with your own 
                customized network architecture. It must however take the same 
                arguments as `NERDANetwork`.
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
        else:
            self.device = device
            print("Device set to:", self.device)
        self.tag_scheme = tag_scheme
        self.tag_outside = tag_outside
        self.transformer = transformer  
        self.dataset_training = dataset_training
        self.dataset_validation = dataset_validation
        self.hyperparameters = hyperparameters
        self.tag_outside = tag_outside
        self.tag_scheme = tag_scheme
        tag_complete = [tag_outside] + tag_scheme
        # fit encoder to _all_ possible tags.
        self.max_len = max_len
        self.tag_encoder = sklearn.preprocessing.LabelEncoder()
        self.tag_encoder.fit(tag_complete)
        self.transformer_model = AutoModel.from_pretrained(transformer)
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(transformer, **tokenizer_parameters)
        self.transformer_config = AutoConfig.from_pretrained(transformer)  
        self.network = NERDANetwork(self.transformer_model, self.device, len(tag_complete), dropout = dropout)
        self.network.to(self.device)
        self.validation_batch_size = validation_batch_size
        self.num_workers = num_workers
        self.train_losses = []
        self.valid_loss = np.nan
        self.quantized = False
        self.halved = False

    def train(self) -> str:
        """Train Network

        Trains the network from the NERDA model specification.

        Returns:
            str: a message saying if the model was trained succesfully.
            The network in the 'network' attribute is trained as a 
            side-effect. Training losses and validation loss are saved 
            in 'training_losses' and 'valid_loss' 
            attributes respectively as side-effects.
        """
        network, train_losses, valid_loss = train_model(network = self.network,
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
        setattr(self, "train_losses", train_losses)
        setattr(self, "valid_loss", valid_loss)

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
        self.network.load_state_dict(torch.load(model_path, map_location = torch.device(self.device)))
        return f'Weights for network loaded from {model_path}'

    def quantize(self):
        """Apply dynamic quantization to increase performance.

        Quantization and half precision inference are mutually exclusive.

        Read more: https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html

        Returns:
            Nothing. Applies dynamic quantization to Network as a side-effect.
        """
        assert not (self.quantized), "Dynamic quantization already applied"
        assert not (self.halved), "Can't run both quantization and half precision"

        self.network = torch.quantization.quantize_dynamic(
            self.network, {torch.nn.Linear}, dtype=torch.qint8
        )
        self.quantized = True

    def half(self):
        """Convert weights from Float32 to Float16 to increase performance

        Quantization and half precision inference are mutually exclusive.

        Read more: https://pytorch.org/docs/master/generated/torch.nn.Module.html?highlight=half#torch.nn.Module.half

        Returns: 
            Nothing. Model is "halved" as a side-effect.
        """
        assert not (self.halved), "Half precision already applied"
        assert not (self.quantized), "Can't run both quantization and half precision"

        self.network.half()
        self.halved = True

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
            tuple: word-tokenized sentences and predicted 
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

    def evaluate_performance(self, dataset: dict, 
                             return_accuracy: bool=False,
                             **kwargs) -> pd.DataFrame:
        """Evaluate Performance

        Evaluates the performance of the model on an arbitrary
        data set.

        Args:
            dataset (dict): Data set that must consist of
                'sentences' and NER'tags'. You can look at examples
                 of, how the dataset should look like by invoking functions 
                 get_dane_data() or get_conll_data().
            kwargs: arbitrary keyword arguments for predict. For
                instance 'batch_size' and 'num_workers'.
            return_accuracy (bool): Return accuracy
                as well? Defaults to False.

            
        Returns:
            DataFrame with performance numbers, F1-scores,
            Precision and Recall. Returns dictionary with
            this AND accuracy, if return_accuracy is set to
            True.
        """
        
        tags_predicted = self.predict(dataset.get('sentences'), 
                                      **kwargs)
        
        # compute F1 scores by entity type
        f1 = compute_f1_scores(y_pred = tags_predicted, 
                               y_true = dataset.get('tags'),
                               labels = self.tag_scheme,
                               average = None)
        
        # create DataFrame with performance scores (=F1)
        df = list(zip(self.tag_scheme, f1[2], f1[0], f1[1]))
        df = pd.DataFrame(df, columns = ['Level', 'F1-Score', 'Precision', 'Recall'])    
        
        # compute MICRO-averaged F1-scores and add to table.
        f1_micro = compute_f1_scores(y_pred = tags_predicted, 
                                     y_true = dataset.get('tags'),
                                     labels = self.tag_scheme,
                                     average = 'micro')
        f1_micro = pd.DataFrame({'Level' : ['AVG_MICRO'], 
                                 'F1-Score': [f1_micro[2]],
                                 'Precision': [np.nan],
                                 'Recall': [np.nan]})
        df = df.append(f1_micro)

        # compute MACRO-averaged F1-scores and add to table.
        f1_macro = compute_f1_scores(y_pred = tags_predicted, 
                                     y_true = dataset.get('tags'),
                                     labels = self.tag_scheme,
                                     average = 'macro')
        f1_macro = pd.DataFrame({'Level' : ['AVG_MICRO'], 
                                 'F1-Score': [f1_macro[2]],
                                 'Precision': [np.nan],
                                 'Recall': [np.nan]})
        df = df.append(f1_macro)

        # compute and return accuracy if desired
        if return_accuracy:
            accuracy = accuracy_score(y_pred = flatten(tags_predicted), 
                                      y_true = flatten(dataset.get('tags')))
            return {'f1':df, 'accuracy': accuracy}
      
        return df

