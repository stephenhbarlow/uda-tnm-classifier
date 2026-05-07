import argparse
import numpy as np
import torch
from torch.optim import AdamW
import random
from models.multi_label_model import MultiLabelModel
from transformers import get_linear_schedule_with_warmup
from trainer.multi_label_trainer import MultiLabelTrainer
from data_loaders.data_loaders import MultiLabelDataLoader
from torch.utils.tensorboard import SummaryWriter
import os

# Disable tokenizer parallelism to get rid of annoying warning.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--train_data_dir', type=str, default='data/tnm_inc_train.csv',
                        help='the path to the directory containing the training data.')
    parser.add_argument('--val_data_dir', type=str, default='data/tnm_inc_val.csv',
                        help='the path to the directory containing the validation data.')
    parser.add_argument('--tokenizer', type=str, default="da_gatortron_base_GSTTv2/results",
                        help='the pretrained tokenizer.')

    # Data loader settings
    parser.add_argument('--max_len', type=int, default=512, help='max length of sentence encoding')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=8, help='the number of samples for a batch')

    # Model settings (for Transformer)
    parser.add_argument('--model_ckpt', type=str, default="da_gatortron_base_GSTTv2/results",
                        help='the pretrained Transformer.')
    parser.add_argument('--n_classes', type=int, default=4, help='the number of output classes')
    parser.add_argument('--dropout_prob', type=float, default=0.1, help='the dropout rate of the output layer.')

    # Trainer settings
    parser.add_argument('--device', type=str, default='cuda', help="'mps' or 'cuda'")
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=5, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/',
                        help='the patch to save the results of experiments.')
    parser.add_argument('--log_period', type=int, default=1000, help='the logging interval (in batches).')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period (in epochs).')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'],
                        help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='Acc', help='the metric to be monitored.')

    # Optimization
    parser.add_argument('--lr', type=float, default=1e-5, help='the starting learning rate.')

    # Others
    parser.add_argument('--seed', type=int, default=42, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    parser.add_argument('--text_num_protype', type=int, default=10, help='.')
    
    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_args()
    
    # fix random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # create data loader
    train_dataloader = MultiLabelDataLoader(args, split='train', shuffle=True)
    val_dataloader = MultiLabelDataLoader(args, split='val', shuffle=False)
    # test_dataloader = BertDataLoader(args, split='test', shuffle=False)

    # build model architecture
    model = MultiLabelModel(args)

    # build optimizer, learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dataloader) * args.epochs
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_steps,
                                                   num_training_steps=total_steps)

    # build trainer and start to train
    trainer = MultiLabelTrainer(model, optimizer, args, lr_scheduler, train_dataloader, val_dataloader)
    trainer.train()

if __name__ == '__main__':
    main()