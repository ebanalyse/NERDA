import torch

class DataSetReaderNER():
    def __init__(self, sentences, tags, transformer_tokenizer, max_len, tag_encoder):
        self.sentences = sentences
        self.tags = tags
        self.transformer_tokenizer = transformer_tokenizer
        self.max_len = max_len
        self.tag_encoder = tag_encoder

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
        
        for i, word in enumerate(sentence):
            # bert tokenization
            wordpieces = self.transformer_tokenizer.tokenize(word)
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
        input_ids = self.transformer_tokenizer.encode(tokens)

        # fill out other inputs for model.    
        # 8 is the 'O' encoding
        target_tags = [8] + target_tags + [8] 
        masks = [1] * len(input_ids)
        # set to 0, because we are not doing NSP or QA type task (across multiple sentences)
        # token_type_ids distinguishes sentences.
        token_type_ids = [0] * len(input_ids) 
        offsets = [1] + offsets + [1]

        # Padding to max length 
        # compute padding length
        padding_len = self.max_len - len(input_ids)
        input_ids = input_ids + ([0] * padding_len)
        masks = masks + ([0] * padding_len)  
        offsets = offsets + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        # set to 8, since 'O' encoded as 8
        target_tags = target_tags + ([8] * padding_len)  

        return {'input_ids' : torch.tensor(input_ids, dtype = torch.long),
                'masks' : torch.tensor(masks, dtype = torch.long),
                'token_type_ids' : torch.tensor(token_type_ids, dtype = torch.long),
                'target_tags' : torch.tensor(target_tags, dtype = torch.long),
                'offsets': offsets} 
      
def create_dataloader(sentences, tags, transformer_tokenizer, max_len, batch_size, tag_encoder, num_workers = 1):
    
    data_reader = DataSetReaderNER(
        sentences = sentences, 
        tags = tags,
        transformer_tokenizer = transformer_tokenizer, 
        max_len = max_len,
        tag_encoder = tag_encoder)

    data_loader = torch.utils.data.DataLoader(
        data_reader, batch_size = batch_size, num_workers = num_workers
    )

    return data_loader

