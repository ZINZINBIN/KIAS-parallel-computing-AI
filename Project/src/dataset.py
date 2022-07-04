import torch
import torch.nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import random
import re

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer : BertTokenizer, max_len : int, max_pred : int, mask_prob : int):
        super().__init__()
        self.df = df # data
        self.len = len(self.df)
        self.max_pred = max_pred # max tokens of prediction
        self.mask_prob = mask_prob # masking probability
        self.tokenizer = tokenizer # tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        item = self.process(item)
        return item
           
    def process(self, row):
        seq = row['sequence'] # 'ACGEF...'
        class_num = row['cls_idx']
        
        # random crop
        if len(seq) >= self.max_len+5:
            idx = random.randint(0,len(seq)-self.max_len-5)
            seq = seq[idx:idx+self.max_len]
        seq = re.sub(r"[UZOB]", "X", seq)
        tokens = list(seq)

        # Tokenize
        ret = self.tokenizer(" ".join(tokens),
                            return_tensors = 'pt',
                            max_length = self.max_len,
                            padding = 'max_length',
                            truncation=True,
                            add_special_tokens=True)
        ret = {k: v.squeeze() for k,v in ret.items()}
        ret['label'] = torch.tensor(class_num)
        return ret
    
    def __len__(self):
        return len(self.df)