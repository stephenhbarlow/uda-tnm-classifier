import argparse
import numpy as np
import torch
import random
import data.datasets
import trainer.train_utils
from models.metric import display_roc_curve, display_confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from models.multi_label_model import MultiLabelModel
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_dir', type=str, default='data/royal_free_tnm.csv',
                        help='the path to the directory containing the test data.')
    parser.add_argument('--model', type=str, default='results/full_ds_no_tsa/GatorTron_UDATrueLR_1e-05epochs_5 train_data1999 examples_TSA_false_shuffled_findings.bin')
    parser.add_argument('--tokenizer', type=str, default="UFNLP/gatortron-base",
                        help='the pretrained tokenizer.')
    parser.add_argument('--sequence_length', type=int, default=512, help='Sequence length for Bert')

    # Data loader settings
    parser.add_argument('--max_len', type=int, default=512, help='max length of sentence encoding')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=4, help='the number of samples for a batch')

    # Model settings (for Transformer)
    parser.add_argument('--model_ckpt', type=str, default="UFNLP/gatortron-base",
                        help='the pretrained Transformer.')
    parser.add_argument('--n_classes', type=int, default=2, help='the number of output classes')
    parser.add_argument('--dropout_prob', type=float, default=0.1, help='the dropout rate of the output layer.')

    parser.add_argument('--seed', type=int, default=1234, help='.')

    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_args()

    # fix random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    validation_criterion = torch.nn.CrossEntropyLoss()

    # build model architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiLabelModel(args).to(device)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    # create data loader
    test_dataset = data.datasets.SupervisedDataset(args.test_data_dir, args.tokenizer, args.sequence_length, args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    test_dict = trainer.train_utils.eval_model(device, model, test_dataloader, validation_criterion)
    test_acc = accuracy_score(test_dict['y_labels'], test_dict['preds'])
    print(f"Test Accuracy: {test_acc} ")
    print(classification_report(test_dict['y_labels'], test_dict['preds']))
    display_confusion_matrix(test_dict['y_labels'],  test_dict['preds'])
    display_roc_curve(test_dict['y_labels'], test_dict['probs'][:, -1])


if __name__ == '__main__':
    main()