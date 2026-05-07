import torch
import torch.nn.functional as nnf
import numpy as np
from models.metric import display_roc_curve, display_confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, hamming_loss

# not edited yet!!!!!!!!!
class MultiLabelEvaluateModel(object):

    def __init__(self, model, args, test_dataloader):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.args = args
        self.test_dataloader = test_dataloader
        self.loss = torch.nn.BCEWithLogitsLoss()

    def eval_model(self):

        self.model = self.model.eval()
        losses = []
        probs = []
        preds = []
        y_test_labels = []

        with torch.no_grad():
            for d in self.test_dataloader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                labels = d["labels"].to(self.device)
                logits = self.model(input_ids=input_ids,
                                        attention_mask=attention_mask)
                pred = torch.round(torch.sigmoid(logits)).squeeze()
                preds.append(pred)
                prob = torch.sigmoid(logits)
                probs.append(prob)
                y_test_labels.append(labels)
                loss = self.loss(logits, labels.float())
                losses.append(loss.item())
        y_test_labels = torch.cat(y_test_labels).cpu().data.numpy()
        preds = torch.cat(preds).cpu().data.numpy()
        probs = torch.cat(probs).cpu().data.numpy()

        return {
                "loss": np.mean(losses), 
                "y_labels": y_test_labels, 
                "preds": preds, 
                "probs": probs,
                }

    def eval(self):
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