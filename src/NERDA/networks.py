"""Networks for NERDA"""

import torch.nn as nn
from transformers import AutoConfig
from .utils import match_kwargs

class NERDANetwork(nn.Module):
    """A Generic Network for NERDA models.

    The network has the same architecture as the models in
    [Hvingelby et al. 2020](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.565.pdf).

    Can be replaced with a custom user-defined network with 
    the restriction, that it must take the arguments.
    """

    def __init__(self, transformer, device, n_tags, dropout = 0.1) -> None:
        """Initialize NERDA network"""
        super(NERDANetwork, self).__init__()
        
        # extract transformer name
        transformer_name = transformer.name_or_path
        # extract AutoConfig, from which relevant parameters can be extracted.
        transformer_config = AutoConfig.from_pretrained(transformer_name)

        self.transformer = transformer
        self.dropout = nn.Dropout(dropout)
        self.tags = nn.Linear(transformer_config.hidden_size, n_tags)
        self.device = device

    # NOTE: offsets are not used in model as-is, but they are expected as output
    # down-stream. So _DON'T_ remove! :)
    def forward(self, input_ids, masks, token_type_ids, target_tags, offsets):

        # TODO: can be improved with ** and move everything to device in a
        # single step.
        transformer_inputs = {
            'input_ids': input_ids.to(self.device),
            'masks': masks.to(self.device),
            'token_type_ids': token_type_ids.to(self.device)
            }
        
        # match args with transformer
        transformer_inputs = match_kwargs(self.transformer.forward, **transformer_inputs)
           
        outputs = self.transformer(**transformer_inputs)[0]

        # apply drop-out
        outputs = self.dropout(outputs)

        # outputs for all labels/tags
        outputs = self.tags(outputs)

        return outputs

