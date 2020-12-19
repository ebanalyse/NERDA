import numpy as np
from torch.utils.tensorboard import SummaryWriter

dabert = 'DJSammy/bert-base-danish-uncased_BotXO,ai'

# Do not include the 'O' (not entity) tag when computing loss function
writer = SummaryWriter('runs/tensorboard')