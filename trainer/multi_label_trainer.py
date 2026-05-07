from tqdm import tqdm
import torch
import logging
import os
import numpy as np
from models.metric import display_roc_curve, display_confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, hamming_loss


class MultiLabelTrainer(object):

    def __init__(self, model, optimizer, args, lr_scheduler, train_dataloader, val_dataloader):

        self.args = args
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model onto configured device
        if self.args.device == "mps":
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        elif self.args.device == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = self.args.epochs
        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir
        self.loss = torch.nn.BCEWithLogitsLoss()
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def train_epoch(self):
        progress = tqdm(enumerate(self.train_dataloader),
                        total=len(self.train_dataloader))
        self.model = self.model.train()
        losses = []
        predictions = []
        labels = []
        for _, d in progress:
            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            labels = d["labels"].to(self.device)
            logits = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask)
            preds = torch.round(torch.sigmoid(logits)).squeeze()
            predictions.append(preds)
            loss = self.loss(logits, labels.float())
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        return np.mean(losses)

    def eval_model(self):

        self.model = self.model.eval()
        losses = []
        probs = []
        predictions = []
        y_test_labels = []

        with torch.no_grad():
            for d in self.val_dataloader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                labels = d["labels"].to(self.device)
                logits = self.model(input_ids=input_ids,
                                        attention_mask=attention_mask)
                preds = torch.round(torch.sigmoid(logits))
                predictions.append(preds)
                prob = torch.sigmoid(logits)
                probs.append(prob)
                y_test_labels.append(labels)
                loss = self.loss(logits, labels.float())
                losses.append(loss.item())
        y_test_labels = torch.cat(y_test_labels).cpu().data.numpy()
        predictions = torch.cat(predictions).squeeze().cpu().data.numpy()
        probs = torch.cat(probs).cpu().data.numpy()

        return {
                "loss": np.mean(losses), 
                "y_labels": y_test_labels, 
                "preds": predictions, 
                "probs": probs,
                }

    def train(self):
        best_hamming_dist = 1
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            print('-' * 10)
            train_loss = self.train_epoch()
            print(f'Train Loss: {train_loss}')
            eval_dict = self.eval_model()
            abs_acc = accuracy_score(eval_dict['y_labels'], eval_dict['preds'])
            hamming_dist = hamming_loss(eval_dict['y_labels'], eval_dict['preds'])
            t_acc = accuracy_score(eval_dict['y_labels'][:, 0], eval_dict['preds'][:, 0])
            n_acc = accuracy_score(eval_dict['y_labels'][:, 1], eval_dict['preds'][:, 1])
            m_acc = accuracy_score(eval_dict['y_labels'][:, 2], eval_dict['preds'][:, 2])
            u_acc = accuracy_score(eval_dict['y_labels'][:, 3], eval_dict['preds'][:, 3])
            print(f"Val Loss: {eval_dict['loss']} "
                  f"Absolute Acc: {abs_acc} "
                  f"Hamming Distance: {hamming_dist} "
                  f"Tumour Acc: {t_acc} "
                  f"Node Acc: {n_acc} "
                  f"Mets Acc: {m_acc} "
                  f"Uncertainty Acc: {u_acc} ")
 
            # if hamming_dist < best_hamming_dist:
            torch.save(self.model.state_dict(),
                        f"{self.checkpoint_dir}/DAmultilabel{self.args.train_data_dir[5:11]}"
                        f"{self.args.n_classes}classes{epoch + 1}epochs.bin")
            best_hamming_dist = hamming_dist

        # print(compute_scores(predictions, y_test))
        print(classification_report(eval_dict['y_labels'][:, 0], eval_dict['preds'][:, 0]))
        print(classification_report(eval_dict['y_labels'][:, 1], eval_dict['preds'][:, 1]))
        print(classification_report(eval_dict['y_labels'][:, 2], eval_dict['preds'][:, 2]))
        print(classification_report(eval_dict['y_labels'][:, 3], eval_dict['preds'][:, 3]))

        display_confusion_matrix(eval_dict['y_labels'][:, 0], eval_dict['preds'][:, 0])
        display_confusion_matrix(eval_dict['y_labels'][:, 1], eval_dict['preds'][:, 1])
        display_confusion_matrix(eval_dict['y_labels'][:, 2], eval_dict['preds'][:, 2])
        display_confusion_matrix(eval_dict['y_labels'][:, 3], eval_dict['preds'][:, 3])
 
        display_roc_curve(eval_dict['y_labels'][:, 0], eval_dict['probs'][:, 0])
        display_roc_curve(eval_dict['y_labels'][:, 1], eval_dict['probs'][:, 1])
        display_roc_curve(eval_dict['y_labels'][:, 2], eval_dict['probs'][:, 2])
        display_roc_curve(eval_dict['y_labels'][:, 3], eval_dict['probs'][:, 3])