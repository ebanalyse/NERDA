import torch
import numpy as np
import random
from transformers import BertModel, BertTokenizer
from torch.utils.tensorboard import SummaryWriter

dabert = 'DJSammy/bert-base-danish-uncased_BotXO,ai'

# Do not include the 'O' (not entity) tag when computing loss function
writer = SummaryWriter('runs/tensorboard')

def get_bert_model(model_name):
    model = BertModel.from_pretrained(model_name)
    return model

def get_bert_tokenizer(model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case = True)  
    return tokenizer