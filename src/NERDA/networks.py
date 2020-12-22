import torch.nn as nn
from transformers import AutoConfig

class GenericNetwork(nn.Module):
    def __init__(self, transformer, device, n_tags, dropout = 0.1):
        super(GenericNetwork, self).__init__()
        
        # extract transformer name
        transformer_name = transformer.name_or_path
        # extract AutoConfig, from which relevant parameters can be extracted.
        transformer_config = AutoConfig.from_pretrained(transformer_name)

        self.transformer = transformer 
        self.dropout = nn.Dropout(dropout)
        self.tags = nn.Linear(transformer_config.hidden_size, n_tags)
        self.device = device

    # NOTE: offsets not used as-is, but is expected as output
    # down-stream. So don't remove! :)
    def forward(self, input_ids, masks, token_type_ids, target_tags, offsets):

        # move tensors to device
        input_ids = input_ids.to(self.device)
        masks = masks.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        target_tags = target_tags.to(self.device)
        
        # get transformer hidden states
        outputs = self.transformer(input_ids, attention_mask = masks, token_type_ids = token_type_ids)[0]

        # apply drop-out
        outputs = self.dropout(outputs)

        # outputs for all labels/tags
        outputs = self.tags(outputs)

        return outputs