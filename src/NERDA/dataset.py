from .utils import get_bert_tokenizer
import torch

# NOTE: er dette specifikt for DaNE data eller er det en generel DataSet Reader?
class DataSetReaderNER():
    def __init__(self, sentences, ner_tags, bert_model_name, max_len):
        self.sentences = sentences
        self.ner_tags = ner_tags
        self.bert_tokenizer = get_bert_tokenizer(bert_model_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = self.sentences[item]
        tags = self.ner_tags[item]

        # check inputs for consistancy
        assert len(sentence) == len(tags)

        ids = []
        target_tags = []
        tokens = []
        offsets = []
        
        for i, word in enumerate(sentence):
            # bert tokenization.
            wordpieces = self.bert_tokenizer.tokenize(word)
            tokens.extend(wordpieces)
            # make room for CLS
            offsets.extend([1]+[0]*(len(wordpieces)-1))
            # Extends the ner_tag if the word has been split by the wordpiece tokenizer
            target_tags.extend([tags[i]] * len(wordpieces)) 
        
        # Make room for adding special tokens (one for both 'CLS' and 'SEP' special tokens)
        # max_len includes _all_ tokens.
        tokens = tokens[:self.max_len - 2] 
        target_tags = target_tags[:self.max_len - 2]
        offsets = offsets[:self.max_len - 2]

        # encode tokens for BERT
        ids = self.bert_tokenizer.encode(tokens)

        # fill out other inputs for model.    
        # 8 is the 'O' encoding
        target_tags = [8] + target_tags + [8] 
        masks = [1] * len(ids)
        # set to 0, because we are not doing NSP or QA type task (across multiple sentences)
        # token_type_ids distinguishes sentences.
        token_type_ids = [0] * len(ids) 
        offsets = [1] + offsets + [1]

        # Padding to max length 
        # compute padding length
        padding_len = self.max_len - len(ids)
        ids = ids + ([0] * padding_len)
        masks = masks + ([0] * padding_len)  
        offsets = offsets + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        # set to 8, since 'O' encoded as 8
        target_tags = target_tags + ([8] * padding_len)  

        return {'ids' : torch.tensor(ids, dtype = torch.long),
                'masks' : torch.tensor(masks, dtype = torch.long),
                'token_type_ids' : torch.tensor(token_type_ids, dtype = torch.long),
                'target_tags' : torch.tensor(target_tags, dtype = torch.long),
                'offsets': offsets} 
      
      
def create_dataloader(df, bert_model_name, max_len, batch_size):
    
    data_reader = DataSetReaderNER(
        sentences = df['words'].tolist(), 
        ner_tags = df['tags'].tolist(), 
        bert_model_name = bert_model_name, 
        max_len = max_len)

    data_loader = torch.utils.data.DataLoader(
        data_reader, batch_size = batch_size, num_workers = 1
    )

    return data_reader, data_loader

