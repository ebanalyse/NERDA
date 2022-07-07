"""This section covers `torch` networks for `NERDA`"""
import torch
import torch.nn as nn
from transformers import AutoConfig
from NERDA.utils import match_kwargs
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


class TransformerLstmCRF(nn.Module):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, transformer: nn.Module, num_labels: int, dropout: float):
        super(TransformerLstmCRF, self).__init__()

        # extract transformer name
        transformer_name = transformer.name_or_path
        # extract AutoConfig, from which relevant parameters can be extracted.
        transformer_config = AutoConfig.from_pretrained(transformer_name)

        self.num_labels = num_labels

        self.transformer = transformer
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(transformer_config.hidden_size, (transformer_config.hidden_size) // 2, dropout=dropout, batch_first=True,
                              bidirectional=True)
        self.classifier = nn.Linear(
            transformer_config.hidden_size, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        lstm_output, hc = self.bilstm(sequence_output)
        logits = self.classifier(lstm_output)

        loss = None
        if labels is not None:
            log_likelihood, tags = self.crf(
                logits, labels), self.crf.decode(logits)
            loss = 0 - log_likelihood
        else:
            tags = self.crf.decode(logits)
        tags = torch.Tensor(tags)

        if not return_dict:
            output = (tags,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return tags
