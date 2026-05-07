import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import pandas as pd
    

class SupervisedDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, args, n_examples=None):
        super(SupervisedDataset, self).__init__()

        self.data = pd.read_csv(data)
        self.data = self.data.sample(n=n_examples, random_state=args.seed) if n_examples is not None else self.data
        self.len = len(self.data)
        # self.attributes = ['tumour', 'node', 'metastasis', 'uncertainty']
        self.labels = self.data['metastasis'].to_numpy()
        self.attributes = 'metastasis'        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, truncation_side='left')
        self.max_len = max_len

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        row = self.data.iloc[index]
        labels = torch.tensor(self.labels[index], dtype=torch.long)
        text = str(row["text"])
        encoding = self.tokenizer(text,
                                  add_special_tokens=True,
                                  max_length=self.max_len,
                                  return_token_type_ids=False,
                                  truncation=True,
                                  padding='max_length',
                                  return_attention_mask=True,
                                  return_tensors='pt',
                                  )

        encoding['input_ids'] = encoding['input_ids'].flatten()
        encoding['attention_mask'] = encoding['attention_mask'].flatten()

        return encoding, labels


class UnsupervisedDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        super(UnsupervisedDataset, self).__init__()

        self.data = pd.read_csv(data)
        self.len = len(self.data)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, truncation_side='left')
        self.max_len = max_len

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        row = self.data.iloc[index]
        original = str(row['clean text'])
        augmented = str(row['augmented_findings_text'])
        # original = str(row['Report text'])
        # augmented = str(row['clean text'])
        ori_encoding = self.tokenizer(original,
                                      add_special_tokens=True,
                                      max_length=self.max_len,
                                      return_token_type_ids=False,
                                      truncation=True,
                                      padding='max_length',
                                      return_attention_mask=True,
                                      return_tensors='pt',
                                      )
        aug_encoding = self.tokenizer(augmented,
                                      add_special_tokens=True,
                                      max_length=self.max_len,
                                      return_token_type_ids=False,
                                      truncation=True,
                                      padding='max_length',
                                      return_attention_mask=True,
                                      return_tensors='pt',
                                      )

        ori_encoding['input_ids'] = ori_encoding['input_ids'].flatten()
        ori_encoding['attention_mask'] = ori_encoding['attention_mask'].flatten()
        aug_encoding['input_ids'] = aug_encoding['input_ids'].flatten()
        aug_encoding['attention_mask'] = aug_encoding['attention_mask'].flatten()     
        assert ori_encoding['input_ids'].shape == aug_encoding['input_ids'].shape

        return ori_encoding, aug_encoding


class InferenceDataset(Dataset):
    
    def __init__(self, text, tokenizer, max_len):
        self.text = text
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, truncation_side='left')
        self.max_len = max_len
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, item):
        text = str(self.text[item])
        encoding = self.tokenizer(text, 
                                  add_special_tokens=True, 
                                  max_length=self.max_len, 
                                  return_token_type_ids=False,
                                  truncation=True,
                                  padding='max_length',
                                  return_attention_mask=True,
                                  return_tensors='pt')
        
        return {'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()}
    
