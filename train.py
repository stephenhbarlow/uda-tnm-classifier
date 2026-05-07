import os
import torch
import numpy as np
from transformers import get_linear_schedule_with_warmup
import random
from torch.utils.data import DataLoader
from models.metric import display_roc_curve, display_confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from models.multi_label_model import MultiLabelModel
import data.datasets
import trainer.train_utils
import argparse
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # UDA settings
    parser.add_argument('--name', type=str, default='untitled', help='Experiment name')
    parser.add_argument('--prepare_data', action='store_true', help='Prepare data')
    parser.add_argument('--sequence_length', type=int, default=512, help='Sequence length for Bert')
    parser.add_argument('--total_steps', type=int, default=1700, help='Total number of training steps')
    parser.add_argument('--batch_size', type=int, default=8, help='Supervised batch size')
    parser.add_argument('--use_uda', type=bool, default=False, help='Whether to use unsupervised data augmentation')
    parser.add_argument('--unsupervised_ratio', type=float, default=7, help='Ratio of unsupervised batch size to supervised')
    parser.add_argument('--tsa', type=bool, default=False, help="Whether to use Training Signal Annealing")
    parser.add_argument('--uda_coefficient', type=float, default=1, help='Weight for unsupervised loss')
    parser.add_argument('--schedule', type=str, default='linear', help='Schedule for TSA; choose between linear, log, exp')
    parser.add_argument('--uda_softmax_temperature', type=float, default=-1, help='UDA softmax temperature')
    parser.add_argument('--uda_confidence_threshold', type=float, default=-1, help='UDA confidence threshold')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--evaluate_every', type=int, default=334, help='Evaluate on the test set every [_] steps')

    # Data input settings
    parser.add_argument('--train_data_dir', type=str, default='data/tnm_inc_train.csv',
                        help='the path to the directory containing the training data.')
    parser.add_argument('--train_data_examples', type=int, default=1999) 
    parser.add_argument('--val_data_dir', type=str, default='data/tnm_inc_val.csv',
                        help='the path to the directory containing the validation data.')
    parser.add_argument('--unsupervised_data_dir', type=str, default='data/gstt_all_aug_all2.csv',
                        help='the path to the directory containing the validation data.')
    parser.add_argument('--tokenizer', type=str, default="UFNLP/gatortron-base",
                        help='the pretrained tokenizer.')

    # Model settings (for Transformer)
    parser.add_argument('--model_ckpt', type=str, default="UFNLP/gatortron-base",
                        help='the pretrained Transformer.')
    parser.add_argument('--n_classes', type=int, default=2, help='the number of output classes')
    parser.add_argument('--dropout_prob', type=float, default=0.1, help='the dropout rate of the output layer.')

    # Trainer settings
    parser.add_argument('--device', type=str, default='cuda', help="'mps' or 'cuda'")
    parser.add_argument('--epochs', type=int, default=5, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/',
                        help='the patch to save the results of experiments.')
    parser.add_argument('--accumulation', type=int, default=1, help='Number of gradient accumulation steps')

    args = parser.parse_args()

    # fix random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not torch.cuda.is_available():
        print('GPU not available. Running on CPU...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiLabelModel(args).to(device)

    supervised_train_dataset = data.datasets.SupervisedDataset(args.train_data_dir, args.tokenizer, args.sequence_length)
    supervised_validation_dataset = data.datasets.SupervisedDataset(args.val_data_dir, args.tokenizer, args.sequence_length)
    unsupervised_dataset = data.datasets.UnsupervisedDataset(args.unsupervised_data_dir, args.tokenizer, args.sequence_length)

    supervised_train_dataloader = DataLoader(supervised_train_dataset, batch_size=args.batch_size, shuffle=True)
    supervised_validation_dataloader = DataLoader(supervised_validation_dataset, batch_size=args.batch_size, shuffle=False)
    unsupervised_dataloader = DataLoader(unsupervised_dataset, batch_size=args.batch_size * args.unsupervised_ratio, shuffle=True, drop_last=True)
    print(len(supervised_train_dataloader))

    supervised_train_dataiter = iter(supervised_train_dataloader)
    unsupervised_dataiter = iter(unsupervised_dataloader)

    supervised_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    unsupervised_criterion = torch.nn.KLDivLoss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    NUM_ACCUMULATION_STEPS = args.accumulation

    best_validation_accuracy = 0

    total_steps = (args.epochs * len(supervised_train_dataloader)) + 1
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_steps,
                                                   num_training_steps=total_steps)
    
    steps = tqdm(range(1, total_steps + 1))
    for step in steps:
        steps.set_description(f'Best validation accuracy: {best_validation_accuracy:.3f}')
        try:
            supervised_batch = next(supervised_train_dataiter)
        except StopIteration:
            supervised_train_dataiter = iter(supervised_train_dataloader)
            supervised_batch = next(supervised_train_dataiter)

        try:
            unsupervised_batch = next(unsupervised_dataiter)
        except StopIteration:
            unsupervised_dataiter = iter(unsupervised_dataloader)
            unsupervised_batch = next(unsupervised_dataiter)

        if args.use_uda:
            total_loss, supervised_loss, unsupervised_loss = trainer.train_utils.compute_uda_loss(device, model, supervised_batch,
                                                                                    unsupervised_batch,
                                                                                    supervised_criterion,
                                                                                    unsupervised_criterion, step, args)
        else:
            total_loss = trainer.train_utils.compute_supervised_loss(device, model, supervised_batch, supervised_criterion, step,
                                                                     args)

        total_loss = total_loss / NUM_ACCUMULATION_STEPS
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if (step % NUM_ACCUMULATION_STEPS == 0) or step == len(supervised_train_dataloader):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
      
        if not step % len(supervised_train_dataloader):
            eval_dict = trainer.train_utils.eval_model(device, model, supervised_validation_dataloader, supervised_criterion)
            abs_acc = accuracy_score(eval_dict['y_labels'], eval_dict['preds'])
            print(f"Accuracy: {abs_acc}")
            if abs_acc > best_validation_accuracy:
                best_validation_accuracy = abs_acc
                torch.save(model.state_dict(), f"{args.save_dir}/no_UAD_gatortron_{step}steps.bin")

    # print(compute_scores(predictions, y_test))
    print(classification_report(eval_dict['y_labels'], eval_dict['preds']))
    display_confusion_matrix(eval_dict['y_labels'],  eval_dict['preds'])
    display_roc_curve(eval_dict['y_labels'], eval_dict['probs'])
