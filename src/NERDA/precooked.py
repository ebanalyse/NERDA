"""
This sections covers NERDA Models that have been 'precooked' by 
Ekstra Bladet and are publicly available for download.
"""
from .datasets import get_dane_data, get_conll_data
from .models import NERDA
import os
import urllib
from pathlib import Path
from progressbar import ProgressBar

pbar = None

# helper function to show progressbar
def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = ProgressBar(maxval=total_size)

    downloaded = block_num * block_size
    pbar.start()
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

class Precooked(NERDA):
    """Precooked NERDA Model

    NERDA model specification that has been precooked/pretrained
    and is available for download.

    Inherits from [NERDA.models.NERDA][].
    """
    def __init__(self, **kwargs) -> None:
        """Initialize Precooked NERDA Model

        Args:
            kwargs: all arguments for NERDA Model.
        """
        super().__init__(**kwargs)

    def download_network(self, dir = None) -> None:
        """Download Precooked Network from Web

        Args:
            dir (str, optional): Directory where the model file
                will be saved. Defaults to None, in which case
                the model will be saved in a folder '.nerda' in
                your home directory.

        Returns:
            str: Message saying if the download was successfull.
            Model is downloaded as a side-effect.
        """

        model_name = type(self).__name__

        # url for public S3 bucket with NERDA models.
        url_s3 = 'https://nerda.s3-eu-west-1.amazonaws.com'
        url_model = f'{url_s3}/{model_name}.bin'
        
        if dir is None:
            dir = os.path.join(str(Path.home()), '.nerda')

        if not os.path.exists(dir):
            os.mkdir(dir)
            
        file_path = os.path.join(dir, f'{model_name}.bin')
        
        print(
        """
        Please make sure, that you're running the latest version of 'NERDA'
        otherwise the model is not guaranteed to work.
        """
        )
        print(f'Downloading {url_model} to {file_path}')
        urllib.request.urlretrieve(url_model, file_path, show_progress)

        return "Network downloaded successfully. Load network with 'load_network'."

    def load_network(self, file_path: str = None) -> None:
        """Load Pretrained Network

        Loads pretrained network from file.

        Args:
            file_path (str, optional): Path to model file. Defaults to None,
                in which case, the function points to the '.nerda' folder
                the home directory.
        """

        model_name = type(self).__name__
        
        if file_path is None:
            file_path = os.path.join(str(Path.home()), '.nerda', f'{model_name}.bin')

        assert os.path.exists(file_path), "File does not exist! You can download network with download_network()"
        print(
        """
        Model loaded. Please make sure, that you're running the latest version 
        of 'NERDA' otherwise the model is not guaranteed to work.
        """
        )
        self.load_network_from_file(file_path)
        
class DA_BERT_ML(Precooked):
    """NERDA [Multilingual BERT](https://huggingface.co/bert-base-multilingual-uncased) 
    for Danish Finetuned on [DaNE data set](https://github.com/alexandrainst/danlp/blob/master/docs/docs/datasets.md#dane).
    
    Inherits from [NERDA.precooked.Precooked][].
    
    Examples:
        >>> from NERDA.precooked import DA_BERT_ML()
        >>> model = DA_BERT_ML()
        >>> model.download_network()
        >>> model.load_network()
        >>> text = 'Jens Hansen har en bondegård'
        >>> model.predict_text(text)
        ([['Jens', 'Hansen', 'har', 'en', 'bondegård']], [['B-PER', 'I-PER', 'O', 'O', 'O']])
    
    """
    def __init__(self) -> None:
        """Initialize model"""
        super().__init__(transformer = 'bert-base-multilingual-uncased',
                         device = None,
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
                         dataset_training = get_dane_data('train'),
                         dataset_validation = get_dane_data('dev'),
                         max_len = 128,
                         dropout = 0.1,
                         hyperparameters = {'epochs' : 4,
                                            'warmup_steps' : 500,
                                            'train_batch_size': 13,
                                            'learning_rate': 0.0001},
                         tokenizer_parameters = {'do_lower_case' : True})

class DA_ELECTRA_DA(Precooked):
    """NERDA [Danish ELECTRA](https://huggingface.co/Maltehb/-l-ctra-danish-electra-small-uncased) 
    for Danish finetuned on [DaNE data set](https://github.com/alexandrainst/danlp/blob/master/docs/docs/datasets.md#dane).

    Inherits from [NERDA.precooked.Precooked][].

    Examples:
        >>> from NERDA.precooked import DA_ELECTRA_DA()
        >>> model = DA_ELECTRA_DA()
        >>> model.download_network()
        >>> model.load_network()
        >>> text = 'Jens Hansen har en bondegård'
        >>> model.predict_text(text)
        ([['Jens', 'Hansen', 'har', 'en', 'bondegård']], [['B-PER', 'I-PER', 'O', 'O', 'O']])

    """
    def __init__(self) -> None:
        """Initialize model"""
        super().__init__(transformer = 'Maltehb/-l-ctra-danish-electra-small-uncased',
                         device = None,
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
                         dataset_training = get_dane_data('train'),
                         dataset_validation = get_dane_data('dev'),
                         max_len = 128,
                         dropout = 0.1,
                         hyperparameters = {'epochs' : 5,
                                            'warmup_steps' : 500,
                                            'train_batch_size': 13,
                                            'learning_rate': 0.0001},
                         tokenizer_parameters = {'do_lower_case' : True})

class EN_ELECTRA_EN(Precooked):
    """NERDA [English ELECTRA](https://huggingface.co/google/electra-small-discriminator) 
    for English finetuned on [CoNLL-2003 data set](https://www.clips.uantwerpen.be/conll2003/ner/).

    Inherits from [NERDA.precooked.Precooked][].

    Examples:
        >>> from NERDA.precooked import EN_ELECTRA_EN()
        >>> model = EN_ELECTRA_EN()
        >>> model.download_network()
        >>> model.load_network()
        >>> text = 'Old MacDonald had a farm'
        >>> model.predict_text(text)
        ([['Old', 'MacDonald', 'had', 'a', 'farm']], [['B-PER', 'I-PER', 'O', 'O', 'O']])

    """
    def __init__(self) -> None:
        """Initialize model"""
        super().__init__(transformer = 'google/electra-small-discriminator',
                         device = None,
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
                         dataset_training = get_conll_data('train'),
                         dataset_validation = get_conll_data('valid'),
                         max_len = 128,
                         dropout = 0.1,
                         hyperparameters = {'epochs' : 4,
                                            'warmup_steps' : 250,
                                            'train_batch_size': 13,
                                            'learning_rate': 8e-05},
                         tokenizer_parameters = {'do_lower_case' : True})


class EN_BERT_ML(Precooked):
    """NERDA [Multilingual BERT](https://huggingface.co/bert-base-multilingual-uncased) 
    for English finetuned on [CoNLL-2003 data set](https://www.clips.uantwerpen.be/conll2003/ner/).
    
    Inherits from [NERDA.precooked.Precooked][].
    
    Examples:
        >>> from NERDA.precooked import EN_BERT_ML()
        >>> model = EN_BERT_ML()
        >>> model.download_network()
        >>> model.load_network()
        >>> text = 'Old MacDonald had a farm'
        >>> model.predict_text(text)
        ([['Old', 'MacDonald', 'had', 'a', 'farm']], [['B-PER', 'I-PER', 'O', 'O', 'O']])
    
    """
    def __init__(self) -> None:
        """Initialize model"""
        super().__init__(transformer = 'bert-base-multilingual-uncased',
                         device = None,
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
                         dataset_training = get_conll_data('train'),
                         dataset_validation = get_conll_data('valid'),
                         max_len = 128,
                         dropout = 0.1,
                         hyperparameters = {'epochs' : 4,
                                            'warmup_steps' : 500,
                                            'train_batch_size': 13,
                                            'learning_rate': 0.0001},
                         tokenizer_parameters = {'do_lower_case' : True})




