from torch.utils.data import DataLoader
from data.datasets import TNMDataset, MultiLabelDataset


class TNMDataLoader(DataLoader):
    
    def __init__(self, args, split, shuffle, drop_last=False):
        self.args = args
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = args.tokenizer
        self.max_len = args.max_len
        self.split = split
        if self.split == 'train':
            self.dataset = TNMDataset(self.args.train_data_dir, self.tokenizer, self.max_len)
        elif self.split == 'val':
            self.dataset = TNMDataset(self.args.val_data_dir, self.tokenizer, self.max_len)
        else:
            self.dataset = TNMDataset(self.args.test_data_dir, self.tokenizer, self.max_len)

        self.init_kwargs = {
                            'dataset': self.dataset,
                            'batch_size': self.batch_size,
                            'shuffle': self.shuffle,
                            'num_workers': self.num_workers,
                            'drop_last': drop_last
                            }
        
        super().__init__(**self.init_kwargs)


class MultiLabelDataLoader(DataLoader):
    
    def __init__(self, args, split, shuffle, drop_last=False):
        self.args = args
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = args.tokenizer
        self.max_len = args.max_len
        self.split = split
        if self.split == 'train':
            self.dataset = MultiLabelDataset(self.args.train_data_dir, self.tokenizer, self.max_len)
        elif self.split == 'val':
            self.dataset = MultiLabelDataset(self.args.val_data_dir, self.tokenizer, self.max_len)
        else:
            self.dataset = MultiLabelDataset(self.args.test_data_dir, self.tokenizer, self.max_len)

        self.init_kwargs = {
                            'dataset': self.dataset,
                            'batch_size': self.batch_size,
                            'shuffle': self.shuffle,
                            'num_workers': self.num_workers,
                            'drop_last': drop_last
                            }
        
        super().__init__(**self.init_kwargs)
