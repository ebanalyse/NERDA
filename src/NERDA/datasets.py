"""Functionality for loading Named Entity Recognition data sets."""

import os
import pyconll
from io import BytesIO
from pathlib import Path
from typing import Union, List, Dict
from urllib.request import urlopen
from zipfile import ZipFile

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
    
    print(f'Reading {url_zip}')
    with urlopen(url_zip) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(dir_extract)

    return f'archive extracted to {dir_extract}'

def download_dane_data(dir: str = None) -> str:
    """Download DaNE data set.

    Downloads the 'DaNE' data set annotated for Named Entity
    Recognition kindly hosted by [Alexandra Institute](https://github.com/alexandrainst/danlp/blob/master/docs/docs/datasets.md#dane).

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
        >>> get_dane_data('training', limit = 5)

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
    
    out = {'sentences': sentences, 'tags': entities}
    
    return out

