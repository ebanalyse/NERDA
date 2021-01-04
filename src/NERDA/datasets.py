import os
from pathlib import Path
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import pyconll
from typing import Union, List, Dict

def download_unzip(url_zip: str,
                   dir_extract: str):
    
    print(f'Reading {url_zip}')
    with urlopen(url_zip) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(dir_extract)

    return f'archive extracted to {dir_extract}'

def download_dane_data(dir = os.path.join(str(Path.home()), '.dane')):

    return download_unzip(url_zip = 'http://danlp-downloads.alexandra.dk/datasets/ddt.zip',
                          dir_extract = dir)

def get_dane_data(split: str = 'train', 
                  limit: int = None, 
                  dir: str = os.path.join(str(Path.home()), '.dane')) -> list:
    """Get DaNE Data

    Loads one or more data splits from the DaNE ressource.

    Args:
        split (str, optional): Choose which split to load. You can choose from 'train', 'dev' or 'test'. Defaults to "train".
        limit (int, optional): Limit the number of observations to be returned from a given split. Defaults to None.
        dir (str, optional): Directory where data is cached. Defaults to '.dane' folder in home directory.

    Returns: 
        dict: Dictionary with word-tokenized 'sentences' and NER 'tags' in IOB format.
    """
    
    assert isinstance(split, str)
    splits = ['train', 'dev', 'test']
    assert split in splits, f'Choose between the following splits: {splits}'
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

