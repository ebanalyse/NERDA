"""
This section covers functionality for (down)loading Named Entity 
Recognition data sets.
"""

import csv
import os
import pyconll
from io import BytesIO
from itertools import compress
from pathlib import Path
from typing import Union, List, Dict
from urllib.request import urlopen
from zipfile import ZipFile
import ssl

def download_unzip(url_zip: str,
                   dir_extract: str) -> str:
    """Download and unzip a ZIP archive to folder.

    Loads a ZIP file from URL and extracts all of the files to a 
    given folder. Does not save the ZIP file itself.

    Args:
        url_zip (str): URL to ZIP file.
        dir_extract (str): Directory where files are extracted.

    Returns:
        str: a message telling, if the archive was succesfully
        extracted. Obviously the files in the ZIP archive are
        extracted to the desired directory as a side-effect.
    """
    
    # suppress ssl certification
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    print(f'Reading {url_zip}')
    with urlopen(url_zip, context=ctx) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(dir_extract)

    return f'archive extracted to {dir_extract}'

def download_dane_data(dir: str = None) -> str:
    """Download DaNE data set.

    Downloads the 'DaNE' data set annotated for Named Entity
    Recognition developed and hosted by 
    [Alexandra Institute](https://github.com/alexandrainst/danlp/blob/master/docs/docs/datasets.md#dane).

    Args:
        dir (str, optional): Directory where DaNE datasets will be saved. If no directory is provided, data will be saved to a hidden folder '.dane' in your home directory.  
                           
    Returns:
        str: a message telling, if the archive was in fact 
        succesfully extracted. Obviously the DaNE datasets are
        extracted to the desired directory as a side-effect.
    
    Examples:
        >>> download_dane_data()
        >>> download_dane_data(dir = 'DaNE')
        
    """
    # set to default directory if nothing else has been provided by user.
    if dir is None:
        dir = os.path.join(str(Path.home()), '.dane')

    return download_unzip(url_zip = 'http://danlp-downloads.alexandra.dk/datasets/ddt.zip',
                          dir_extract = dir)

def get_dane_data(split: str = 'train', 
                  limit: int = None, 
                  dir: str = None) -> dict:
    """Load DaNE data split.

    Loads a single data split from the DaNE data set kindly hosted
    by [Alexandra Institute](https://github.com/alexandrainst/danlp/blob/master/docs/docs/datasets.md#dane).

    Args:
        split (str, optional): Choose which split to load. Choose 
            from 'train', 'dev' and 'test'. Defaults to 'train'.
        limit (int, optional): Limit the number of observations to be 
            returned from a given split. Defaults to None, which implies 
            that the entire data split is returned.
        dir (str, optional): Directory where data is cached. If set to 
            None, the function will try to look for files in '.dane' folder in home directory.

    Returns:
        dict: Dictionary with word-tokenized 'sentences' and named 
        entity 'tags' in IOB format.

    Examples:
        Get test split
        >>> get_dane_data('test')

        Get first 5 observations from training split
        >>> get_dane_data('train', limit = 5)

    """
    assert isinstance(split, str)
    splits = ['train', 'dev', 'test']
    assert split in splits, f'Choose between the following splits: {splits}'

    # set to default directory if nothing else has been provided by user.
    if dir is None:
        dir = os.path.join(str(Path.home()), '.dane')
    assert os.path.isdir(dir), f'Directory {dir} does not exist. Try downloading DaNE data with download_dane_data()'
    
    file_path = os.path.join(dir, f'ddt.{split}.conllu')
    assert os.path.isfile(file_path), f'File {file_path} does not exist. Try downloading DaNE data with download_dane_data()'
    
    split = pyconll.load_from_file(file_path)

    sentences = []
    entities = []

    for sent in split:
        sentences.append([token.form for token in sent._tokens])
        entities.append([token.misc['name'].pop() for token in sent._tokens])
    
    if limit is not None:
        sentences = sentences[:limit]
        entities = entities[:limit]
    
    return {'sentences': sentences, 'tags': entities}
    
    

def download_conll_data(dir: str = None) -> str:
    """Download CoNLL-2003 English data set.

    Downloads the [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/) 
    English data set annotated for Named Entity Recognition.

    Args:
        dir (str, optional): Directory where CoNLL-2003 datasets will be saved. If no directory is provided, data will be saved to a hidden folder '.dane' in your home directory.  
                           
    Returns:
        str: a message telling, if the archive was in fact 
        succesfully extracted. Obviously the CoNLL datasets are
        extracted to the desired directory as a side-effect.
    
    Examples:
        >>> download_conll_data()
        >>> download_conll_data(dir = 'conll')
        
    """
    # set to default directory if nothing else has been provided by user.
    if dir is None:
        dir = os.path.join(str(Path.home()), '.conll')

    return download_unzip(url_zip = 'https://data.deepai.org/conll2003.zip',
                          dir_extract = dir)

def get_conll_data(split: str = 'train', 
                   limit: int = None, 
                   dir: str = None) -> dict:
    """Load CoNLL-2003 (English) data split.

    Loads a single data split from the 
    [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/) 
    (English) data set.

    Args:
        split (str, optional): Choose which split to load. Choose 
            from 'train', 'valid' and 'test'. Defaults to 'train'.
        limit (int, optional): Limit the number of observations to be 
            returned from a given split. Defaults to None, which implies 
            that the entire data split is returned.
        dir (str, optional): Directory where data is cached. If set to 
            None, the function will try to look for files in '.conll' folder in home directory.

    Returns:
        dict: Dictionary with word-tokenized 'sentences' and named 
        entity 'tags' in IOB format.

    Examples:
        Get test split
        >>> get_conll_data('test')

        Get first 5 observations from training split
        >>> get_conll_data('train', limit = 5)

    """
    assert isinstance(split, str)
    splits = ['train', 'valid', 'test']
    assert split in splits, f'Choose between the following splits: {splits}'

    # set to default directory if nothing else has been provided by user.
    if dir is None:
        dir = os.path.join(str(Path.home()), '.conll')
    assert os.path.isdir(dir), f'Directory {dir} does not exist. Try downloading CoNLL-2003 data with download_conll_data()'
    
    file_path = os.path.join(dir, f'{split}.txt')
    assert os.path.isfile(file_path), f'File {file_path} does not exist. Try downloading CoNLL-2003 data with download_conll_data()'

    # read data from file.
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter = ' ')
        for row in reader:
            data.append([row])

    sentences = []
    sentence = []
    entities = []
    tags = []

    for row in data:
        # extract first element of list.
        row = row[0]
        # TO DO: move to data reader.
        if len(row) > 0 and row[0] != '-DOCSTART-':
            sentence.append(row[0])
            tags.append(row[-1])        
        if len(row) == 0 and len(sentence) > 0:
            # clean up sentence/tags.
            # remove white spaces.
            selector = [word != ' ' for word in sentence]
            sentence = list(compress(sentence, selector))
            tags = list(compress(tags, selector))
            # append if sentence length is still greater than zero..
            if len(sentence) > 0:
                sentences.append(sentence)
                entities.append(tags)
            sentence = []
            tags = []
            
   
    if limit is not None:
        sentences = sentences[:limit]
        entities = entities[:limit]
    
    return {'sentences': sentences, 'tags': entities}

    




    
    



    


