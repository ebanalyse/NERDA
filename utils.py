import torch
import numpy as np
import random
from transformers import BertModel, BertTokenizer
from torch.utils.tensorboard import SummaryWriter

dabert = 'DJSammy/bert-base-danish-uncased_BotXO,ai'

# Do not include the 'O' (not entity) tag when computing loss function
target_tags = ['B-ORG', 'B-LOC', 'B-PER','B-MISC', 'I-PER', 'I-LOC', 'I-MISC', 'I-ORG']
writer = SummaryWriter('runs/tensorboard')

def get_bert_model(model_name):
    model = BertModel.from_pretrained(model_name)
    return model

def get_bert_tokenizer(model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case = True)  
    return tokenizer

def enforce_reproducibility(seed = 42):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For atomic operations there is currently 
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)