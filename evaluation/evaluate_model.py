import torch
import torch.nn.functional as nnf
import numpy as np
from models.metric import display_confusion_matrix, display_roc_curve
from sklearn.metrics import classification_report

# not edited yet!!!!!!!!!
class EvaluateModel(object):

    def __init__(self, model, args, test_dataloader):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = model.to(self.device)
        self.args = args
        self.test_dataloader = test_dataloader
        self.cancer_loss = torch.nn.CrossEntropyLoss()
        self.tumour_loss = torch.nn.CrossEntropyLoss()
        self.node_loss = torch.nn.BCEWithLogitsLoss()
        self.mets_loss = torch.nn.BCEWithLogitsLoss()

    def eval_model(self):

        self.model = self.model.eval()
        losses = []
        correct_c_preds = 0
        correct_t_preds = 0
        correct_n_preds = 0
        correct_m_preds = 0
        y_test_cancer = []
        y_test_tumour = []
        y_test_node = []
        y_test_mets = []
        cancer_preds = []
        cancer_probs = []
        tumour_preds = []
        tumour_probs = []
        node_preds = []
        node_probs = []
        mets_preds = []
        mets_probs = []

        with torch.no_grad():
            for d in self.test_dataloader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                cancer = d["cancer"].to(self.device)
                tumour = d["tumour"].to(self.device)
                node = d["node"].to(self.device)
                mets = d["mets"].to(self.device)
                logit_dict = self.model(input_ids=input_ids,
                                        attention_mask=attention_mask)

                _, c_preds = torch.max(logit_dict["c"], dim=1)
                _, t_preds = torch.max(logit_dict["t"], dim=1)
                n_preds = torch.round(torch.sigmoid(logit_dict["n"])).squeeze()
                m_preds = torch.round(torch.sigmoid(logit_dict["m"])).squeeze()

                cancer_preds.append(c_preds)
                tumour_preds.append(t_preds)
                node_preds.append(n_preds)
                mets_preds.append(m_preds)

                c_prob = nnf.softmax(logit_dict["c"], dim=1)
                t_prob = nnf.softmax(logit_dict["t"], dim=1)
                n_prob = torch.sigmoid(logit_dict["n"])
                m_prob = torch.sigmoid(logit_dict["m"])

                cancer_probs.append(c_prob)
                tumour_probs.append(t_prob)
                node_probs.append(n_prob)
                mets_probs.append(m_prob)

                y_test_cancer.append(cancer)
                y_test_tumour.append(tumour)
                y_test_node.append(node)
                y_test_mets.append(mets)

                cancer_loss = self.cancer_loss(logit_dict["c"], cancer)
                tumour_loss = self.tumour_loss(logit_dict["t"], tumour)
                node_loss = self.node_loss(logit_dict["n"], node.unsqueeze(1).float())
                mets_loss = self.mets_loss(logit_dict["m"], mets.unsqueeze(1).float())

                correct_c_preds += torch.sum(c_preds == cancer)
                correct_t_preds += torch.sum(t_preds == tumour)
                correct_n_preds += torch.sum(n_preds == node)
                correct_m_preds += torch.sum(m_preds == mets)

                # incidental_loss = self.incidental_loss(incidental_logits, incidentals.unsqueeze(1).float())
                loss = cancer_loss + tumour_loss + node_loss + mets_loss
                losses.append(loss.item())

        y_test_cancer = torch.cat(y_test_cancer).cpu().data.numpy()
        y_test_tumour = torch.cat(y_test_tumour).cpu().data.numpy()
        y_test_node = torch.cat(y_test_node).cpu().data.numpy()
        y_test_mets = torch.cat(y_test_mets).cpu().data.numpy()

        cancer_preds = torch.cat(cancer_preds).cpu().data.numpy()
        tumour_preds = torch.cat(tumour_preds).cpu().numpy()
        node_preds = torch.cat(node_preds).cpu().numpy()
        mets_preds = torch.cat(mets_preds).cpu().numpy()

        cancer_probs = torch.cat(cancer_probs).cpu().data.numpy()
        tumour_probs = torch.cat(tumour_probs).cpu().data.numpy()
        node_probs = torch.cat(node_probs).cpu().data.numpy()
        mets_probs = torch.cat(mets_probs).cpu().data.numpy()

        return {"c_acc": correct_c_preds.float() / len(self.test_dataloader.dataset),
                "t_acc": correct_t_preds.float() / len(self.test_dataloader.dataset),
                "n_acc": correct_n_preds.float() / len(self.test_dataloader.dataset),
                "m_acc": correct_m_preds.float() / len(self.test_dataloader.dataset),
                "loss": np.mean(losses), "y_cancer": y_test_cancer, 
                "y_tumour": y_test_tumour, "y_node": y_test_node, "y_mets": y_test_mets,
                "c_preds": cancer_preds, "t_preds": tumour_preds,
                "n_preds": node_preds, "m_preds": mets_preds, "c_probs": cancer_probs,
                "t_probs": tumour_probs, "n_probs": node_probs, "m_probs": mets_probs}

    def eval(self):
        eval_dict = self.eval_model()
        print(f'Val Loss: {eval_dict["loss"]} Cancer Acc: {eval_dict["c_acc"]}'
              f' Tumour Acc: {eval_dict["t_acc"]} Node Acc: {eval_dict["n_acc"]}'
              f' Mets Acc: {eval_dict["m_acc"]}')

        print(classification_report(eval_dict["y_cancer"], eval_dict["c_preds"]))
        print(classification_report(eval_dict["y_tumour"], eval_dict["t_preds"]))
        print(classification_report(eval_dict["y_node"], eval_dict["n_preds"]))
        print(classification_report(eval_dict["y_mets"], eval_dict["m_preds"]))

        display_confusion_matrix(eval_dict["y_cancer"], eval_dict["c_preds"])
        display_confusion_matrix(eval_dict["y_tumour"], eval_dict["t_preds"])
        display_confusion_matrix(eval_dict["y_node"], eval_dict["n_preds"])
        display_confusion_matrix(eval_dict["y_mets"], eval_dict["m_preds"])
 
        # display_roc_curve(eval_dict["y_cancer"], eval_dict["c_probs"])
        # display_roc_curve(eval_dict["y_tumour"], eval_dict["t_probs"])
        display_roc_curve(eval_dict["y_node"], eval_dict["n_probs"])
        display_roc_curve(eval_dict["y_mets"], eval_dict["m_probs"])