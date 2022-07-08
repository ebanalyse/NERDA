"""This section covers `torch` networks for `NERDA`"""
import torch
import torch.nn as nn
from transformers import AutoConfig
from NERDA.utils import match_kwargs
# from transformers.modeling_bert import *
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF

class NERDANetwork(nn.Module):
    """A Generic Network for NERDA models.

    The network has an analogous architecture to the models in
    [Hvingelby et al. 2020](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.565.pdf).

    Can be replaced with a custom user-defined network with 
    the restriction, that it must take the same arguments.
    """

    def __init__(self, transformer: nn.Module, device: str, n_tags: int, dropout: float = 0.1) -> None:
        """Initialize a NERDA Network

        Args:
            transformer (nn.Module): huggingface `torch` transformer.
            device (str): Computational device.
            n_tags (int): Number of unique entity tags (incl. outside tag)
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(NERDANetwork, self).__init__()
        
        # extract transformer name
        transformer_name = transformer.name_or_path
        # extract AutoConfig, from which relevant parameters can be extracted.
        transformer_config = AutoConfig.from_pretrained(transformer_name)

        self.transformer = transformer
        self.dropout = nn.Dropout(dropout)
        self.tags = nn.Linear(transformer_config.hidden_size, n_tags)
        self.device = device

    # NOTE: 'offsets 'are not used in model as-is, but they are expected as output
    # down-stream. So _DON'T_ remove! :)
    def forward(self, 
                input_ids: torch.Tensor, 
                masks: torch.Tensor, 
                token_type_ids: torch.Tensor, 
                target_tags: torch.Tensor, 
                offsets: torch.Tensor) -> torch.Tensor:
        """Model Forward Iteration

        Args:
            input_ids (torch.Tensor): Input IDs.
            masks (torch.Tensor): Attention Masks.
            token_type_ids (torch.Tensor): Token Type IDs.
            target_tags (torch.Tensor): Target tags. Are not used 
                in model as-is, but they are expected downstream,
                so they can not be left out.
            offsets (torch.Tensor): Offsets to keep track of original
                words. Are not used in model as-is, but they are 
                expected as down-stream, so they can not be left out.

        Returns:
            torch.Tensor: predicted values.
        """

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


class BiLSTMCRF(nn.Module):
    """A Generic Network for NERDA models.

    The network has an analogous architecture to the models in
    [Hvingelby et al. 2020](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.565.pdf).

    Can be replaced with a custom user-defined network with 
    the restriction, that it must take the same arguments.
    """

    def __init__(self, transformer: nn.Module, device: str, n_tags: int, dropout: float = 0.1) -> None:
        """Initialize a NERDA Network

        Args:
            transformer (nn.Module): huggingface `torch` transformer.
            device (str): Computational device.
            n_tags (int): Number of unique entity tags (incl. outside tag)
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(BiLSTMCRF, self).__init__()

        # extract transformer name
        transformer_name = transformer.name_or_path
        # extract AutoConfig, from which relevant parameters can be extracted.
        transformer_config = AutoConfig.from_pretrained(transformer_name)

        self.transformer = transformer
        self.dropout = nn.Dropout(dropout)

        self.bilstm = nn.LSTM(
            input_size=768,
            hidden_size=768 // 2,
            batch_first=True,
            num_layers=2,
            dropout=dropout,
            bidirectional=True
        )

        self.classifier = nn.Linear(768, n_tags)
        self.crf = CRF(n_tags, batch_first=True)
        self.device = device

    # NOTE: 'offsets 'are not used in model as-is, but they are expected as output
    # down-stream. So _DON'T_ remove! :)
    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids, input_token_starts = input_data
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]

        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  for layer, starts in zip(sequence_output, input_token_starts)]

        padded_sequence_output = pad_sequence(
            origin_sequence_output, batch_first=True)

        padded_sequence_output = self.dropout(padded_sequence_output)
        lstm_output, _ = self.bilstm(padded_sequence_output)

        logits = self.classifier(lstm_output)

        outputs = (logits,)
        if labels is not None:
            loss_mask = labels.gt(-1)
            loss = self.crf(logits, labels, loss_mask) * (-1)
            outputs = (loss,) + outputs

        # contain: (loss), scores
        return outputs
