import torch
import warnings
import transformers
import sklearn.preprocessing

class NERDADataSetReader():
    """Generic NERDA DataSetReader"""
    
    def __init__(self, 
                sentences: list, 
                tags: list, 
                transformer_tokenizer: transformers.PreTrainedTokenizer, 
                transformer_config: transformers.PretrainedConfig, 
                max_len: int, 
                tag_encoder: sklearn.preprocessing.LabelEncoder, 
                tag_outside: str,
                pad_sequences : bool = True) -> None:
        """Initialize DataSetReader

        Initializes DataSetReader that prepares and preprocesses 
        DataSet for Named-Entity Recognition Task and training.

        Args:
            sentences (list): Sentences.
            tags (list): Named-Entity tags.
            transformer_tokenizer (transformers.PreTrainedTokenizer): 
                tokenizer for transformer.
            transformer_config (transformers.PretrainedConfig): Config
                for transformer model.
            max_len (int): Maximum length of sentences after applying
                transformer tokenizer.
            tag_encoder (sklearn.preprocessing.LabelEncoder): Encoder
                for Named-Entity tags.
            tag_outside (str): Special Outside tag.
            pad_sequences (bool): Pad sequences to max_len. Defaults
                to True.
        """
        self.sentences = sentences
        self.tags = tags
        self.transformer_tokenizer = transformer_tokenizer
        self.max_len = max_len
        self.tag_encoder = tag_encoder
        self.pad_token_id = transformer_config.pad_token_id
        self.tag_outside_transformed = tag_encoder.transform([tag_outside])[0]
        self.pad_sequences = pad_sequences
    
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = self.sentences[item]
        tags = self.tags[item]
        # encode tags
        tags = self.tag_encoder.transform(tags)
        
        # check inputs for consistancy
        assert len(sentence) == len(tags)

        input_ids = []
        target_tags = []
        tokens = []
        offsets = []
        
        # for debugging purposes
        # print(item)
        for i, word in enumerate(sentence):
            # bert tokenization
            wordpieces = self.transformer_tokenizer.tokenize(word)
            tokens.extend(wordpieces)
            # make room for CLS if there is an identified word piece
            if len(wordpieces)>0:
                offsets.extend([1]+[0]*(len(wordpieces)-1))
            # Extends the ner_tag if the word has been split by the wordpiece tokenizer
            target_tags.extend([tags[i]] * len(wordpieces)) 
               
        # Make room for adding special tokens (one for both 'CLS' and 'SEP' special tokens)
        # max_len includes _all_ tokens.
        if len(tokens) > self.max_len-2:
            msg = f'Sentence #{item} length {len(tokens)} exceeds max_len {self.max_len} and has been truncated'
            warnings.warn(msg)
        tokens = tokens[:self.max_len-2] 
        target_tags = target_tags[:self.max_len-2]
        offsets = offsets[:self.max_len-2]

        # encode tokens for BERT
        # TO DO: prettify this.
        input_ids = self.transformer_tokenizer.convert_tokens_to_ids(tokens)
        input_ids = [self.transformer_tokenizer.cls_token_id] + input_ids + [self.transformer_tokenizer.sep_token_id]
        
        # fill out other inputs for model.    
        target_tags = [self.tag_outside_transformed] + target_tags + [self.tag_outside_transformed] 
        masks = [1] * len(input_ids)
        # set to 0, because we are not doing NSP or QA type task (across multiple sentences)
        # token_type_ids distinguishes sentences.
        token_type_ids = [0] * len(input_ids) 
        offsets = [1] + offsets + [1]

        # Padding to max length 
        # compute padding length
        if self.pad_sequences:
            padding_len = self.max_len - len(input_ids)
            input_ids = input_ids + ([self.pad_token_id] * padding_len)
            masks = masks + ([0] * padding_len)  
            offsets = offsets + ([0] * padding_len)
            token_type_ids = token_type_ids + ([0] * padding_len)
            target_tags = target_tags + ([self.tag_outside_transformed] * padding_len)  
    
        return {'input_ids' : torch.tensor(input_ids, dtype = torch.long),
                'masks' : torch.tensor(masks, dtype = torch.long),
                'token_type_ids' : torch.tensor(token_type_ids, dtype = torch.long),
                'target_tags' : torch.tensor(target_tags, dtype = torch.long),
                'offsets': torch.tensor(offsets, dtype = torch.long)} 
      
def create_dataloader(sentences, 
                      tags, 
                      transformer_tokenizer, 
                      transformer_config, 
                      max_len,  
                      tag_encoder, 
                      tag_outside,
                      batch_size = 1,
                      num_workers = 1,
                      pad_sequences = True):

    if not pad_sequences and batch_size > 1:
        print("setting pad_sequences to True, because batch_size is more than one.")
        pad_sequences = True

    data_reader = NERDADataSetReader(
        sentences = sentences, 
        tags = tags,
        transformer_tokenizer = transformer_tokenizer, 
        transformer_config = transformer_config,
        max_len = max_len,
        tag_encoder = tag_encoder,
        tag_outside = tag_outside,
        pad_sequences = pad_sequences)
        # Don't pad sequences if batch size == 1. This improves performance.

    data_loader = torch.utils.data.DataLoader(
        data_reader, batch_size = batch_size, num_workers = num_workers
    )

    return data_loader

