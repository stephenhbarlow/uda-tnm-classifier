from tqdm import tqdm
import torch
import torch.nn.functional as nnf
import logging
import os
import numpy as np
from models.metric import display_roc_curve, display_confusion_matrix
from sklearn.metrics import classification_report


class TNMTrainer(object):

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
        self.uncertainty_loss = torch.nn.BCEWithLogitsLoss()
        self.tumour_loss = torch.nn.BCEWithLogitsLoss()
        self.node_loss = torch.nn.BCEWithLogitsLoss()
        self.mets_loss = torch.nn.BCEWithLogitsLoss()
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def train_epoch(self):
        progress = tqdm(enumerate(self.train_dataloader),
                        total=len(self.train_dataloader))
        self.model = self.model.train()
        losses = []
        correct_u_preds = 0
        correct_t_preds = 0
        correct_n_preds = 0
        correct_m_preds = 0
        correct_utnm_preds = 0
        for _, d in progress:
            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            uncertainty = d["uncertainty"].to(self.device)
            tumour = d["tumour"].to(self.device)
            node = d["node"].to(self.device)
            mets = d["mets"].to(self.device)
            logit_dict = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask)

            u_preds = torch.round(torch.sigmoid(logit_dict["u"])).squeeze()
            t_preds = torch.round(torch.sigmoid(logit_dict["t"])).squeeze()
            n_preds = torch.round(torch.sigmoid(logit_dict["n"])).squeeze()
            m_preds = torch.round(torch.sigmoid(logit_dict["m"])).squeeze()

            correct_u = torch.sum(u_preds == uncertainty)
            correct_u_preds += correct_u
            correct_t = torch.sum(t_preds == tumour)
            correct_t_preds += correct_t
            correct_n = torch.sum(n_preds == node)
            correct_n_preds += correct_n
            correct_m = torch.sum(m_preds == mets)
            correct_m_preds += correct_m
            # correct_utnm_preds += (sum([correct_u, correct_t, correct_n, correct_m]) == 4)
            # correct_u_preds += torch.sum(u_preds == uncertainty)
            # correct_t_preds += torch.sum(t_preds == tumour)
            # correct_n_preds += torch.sum(n_preds == node)
            # correct_m_preds += torch.sum(m_preds == mets)

            u_loss = self.uncertainty_loss(logit_dict["u"], uncertainty.unsqueeze(1).float())
            t_loss = self.tumour_loss(logit_dict["t"], tumour.unsqueeze(1).float())
            n_loss = self.node_loss(logit_dict["n"], node.unsqueeze(1).float())
            m_loss = self.mets_loss(logit_dict["m"], mets.unsqueeze(1).float())
            loss = u_loss + t_loss + n_loss + m_loss
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        return {"u_acc": correct_u_preds.float() / len(self.train_dataloader.dataset),
                "t_acc": correct_t_preds.float() / len(self.train_dataloader.dataset),
                "n_acc": correct_n_preds.float() / len(self.train_dataloader.dataset),
                "m_acc": correct_m_preds.float() / len(self.train_dataloader.dataset),
                "loss": np.mean(losses)}

    def eval_model(self):

        self.model = self.model.eval()
        losses = []
        correct_u_preds = 0
        correct_t_preds = 0
        correct_n_preds = 0
        correct_m_preds = 0
        correct_utnm_preds = 0
        y_test_uncertainty = []
        y_test_tumour = []
        y_test_node = []
        y_test_mets = []
        uncertainty_preds = []
        uncertainty_probs = []
        tumour_preds = []
        tumour_probs = []
        node_preds = []
        node_probs = []
        mets_preds = []
        mets_probs = []

        with torch.no_grad():
            for d in self.val_dataloader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                uncertainty = d["uncertainty"].to(self.device)
                tumour = d["tumour"].to(self.device)
                node = d["node"].to(self.device)
                mets = d["mets"].to(self.device)
                logit_dict = self.model(input_ids=input_ids,
                                        attention_mask=attention_mask)

                u_preds = torch.round(torch.sigmoid(logit_dict["u"])).squeeze()
                t_preds = torch.round(torch.sigmoid(logit_dict["t"])).squeeze()
                n_preds = torch.round(torch.sigmoid(logit_dict["n"])).squeeze()
                m_preds = torch.round(torch.sigmoid(logit_dict["m"])).squeeze()

                uncertainty_preds.append(u_preds)
                tumour_preds.append(t_preds)
                node_preds.append(n_preds)
                mets_preds.append(m_preds)

                u_prob = nnf.softmax(logit_dict["u"], dim=1)
                t_prob = nnf.softmax(logit_dict["t"], dim=1)
                n_prob = torch.sigmoid(logit_dict["n"])
                m_prob = torch.sigmoid(logit_dict["m"])

                uncertainty_probs.append(u_prob)
                tumour_probs.append(t_prob)
                node_probs.append(n_prob)
                mets_probs.append(m_prob)

                y_test_uncertainty.append(uncertainty)
                y_test_tumour.append(tumour)
                y_test_node.append(node)
                y_test_mets.append(mets)

                uncertainty_loss = self.uncertainty_loss(logit_dict["u"], uncertainty.unsqueeze(1).float())
                tumour_loss = self.tumour_loss(logit_dict["t"], tumour.unsqueeze(1).float())
                node_loss = self.node_loss(logit_dict["n"], node.unsqueeze(1).float())
                mets_loss = self.mets_loss(logit_dict["m"], mets.unsqueeze(1).float())

                # correct_u_preds += torch.sum(u_preds == uncertainty)
                # correct_t_preds += torch.sum(t_preds == tumour)
                # correct_n_preds += torch.sum(n_preds == node)
                # correct_m_preds += torch.sum(m_preds == mets)

                correct_u = torch.sum(u_preds == uncertainty)
                correct_u_preds += correct_u
                correct_t = torch.sum(t_preds == tumour)
                correct_t_preds += correct_t
                correct_n = torch.sum(n_preds == node)
                correct_n_preds += correct_n
                correct_m = torch.sum(m_preds == mets)
                correct_m_preds += correct_m
                correct_utnm_preds += (sum([correct_u, correct_t, correct_n, correct_m]) == 4)

                # incidental_loss = self.incidental_loss(incidental_logits, incidentals.unsqueeze(1).float())
                loss = uncertainty_loss + tumour_loss + node_loss + mets_loss
                losses.append(loss.item())

        y_test_uncertainty = torch.cat(y_test_uncertainty).cpu().data.numpy()
        y_test_tumour = torch.cat(y_test_tumour).cpu().data.numpy()
        y_test_node = torch.cat(y_test_node).cpu().data.numpy()
        y_test_mets = torch.cat(y_test_mets).cpu().data.numpy()

        uncertainty_preds = torch.cat(uncertainty_preds).cpu().data.numpy()
        tumour_preds = torch.cat(tumour_preds).cpu().numpy()
        node_preds = torch.cat(node_preds).cpu().numpy()
        mets_preds = torch.cat(mets_preds).cpu().numpy()

        uncertainty_probs = torch.cat(uncertainty_probs).cpu().data.numpy()
        tumour_probs = torch.cat(tumour_probs).cpu().data.numpy()
        node_probs = torch.cat(node_probs).cpu().data.numpy()
        mets_probs = torch.cat(mets_probs).cpu().data.numpy()

        return {"u_acc": correct_u_preds.float() / len(self.val_dataloader.dataset),
                "t_acc": correct_t_preds.float() / len(self.val_dataloader.dataset),
                "n_acc": correct_n_preds.float() / len(self.val_dataloader.dataset),
                "m_acc": correct_m_preds.float() / len(self.val_dataloader.dataset),
                "utnm_acc": correct_utnm_preds.float() / len(self.train_dataloader.dataset),
                "loss": np.mean(losses), "y_uncertainty": y_test_uncertainty, 
                "y_tumour": y_test_tumour, "y_node": y_test_node, "y_mets": y_test_mets,
                "u_preds": uncertainty_preds, "t_preds": tumour_preds,
                "n_preds": node_preds, "m_preds": mets_preds, "u_probs": uncertainty_probs,
                "t_probs": tumour_probs, "n_probs": node_probs, "m_probs": mets_probs}

    def train(self):
        best_avg_accuracy = 0
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            print('-' * 10)
            train_dict = self.train_epoch()
            print(f'Train Loss: {train_dict["loss"]} Uncertainty Acc: {train_dict["u_acc"]}'
                  f' Tumour Acc: {train_dict["t_acc"]} Node Acc: {train_dict["n_acc"]}'
                  f' Mets Acc: {train_dict["m_acc"]}')
            eval_dict = self.eval_model()
            print(f'Val Loss: {eval_dict["loss"]} Uncertainty Acc: {eval_dict["u_acc"]}'
                  f' Tumour Acc: {eval_dict["t_acc"]} Node Acc: {eval_dict["n_acc"]}'
                  f' Mets Acc: {eval_dict["m_acc"]} ')

            avg_val_acc = (eval_dict["u_acc"] + eval_dict["t_acc"] + eval_dict["n_acc"] +
                              eval_dict["m_acc"]) / 4   
            if avg_val_acc > best_avg_accuracy:
                torch.save(self.model.state_dict(),
                           f"{self.checkpoint_dir}/multi{self.args.train_data_dir[5:11]}"
                           f"{self.args.n_classes}classes{epoch + 1}epochs.bin")
                best_avg_accuracy = avg_val_acc

        # print(compute_scores(predictions, y_test))
        print(classification_report(eval_dict["y_uncertainty"], eval_dict["u_preds"]))
        print(classification_report(eval_dict["y_tumour"], eval_dict["t_preds"]))
        print(classification_report(eval_dict["y_node"], eval_dict["n_preds"]))
        print(classification_report(eval_dict["y_mets"], eval_dict["m_preds"]))

        display_confusion_matrix(eval_dict["y_uncertainty"], eval_dict["u_preds"])
        display_confusion_matrix(eval_dict["y_tumour"], eval_dict["t_preds"])
        display_confusion_matrix(eval_dict["y_node"], eval_dict["n_preds"])
        display_confusion_matrix(eval_dict["y_mets"], eval_dict["m_preds"])
 
        display_roc_curve(eval_dict["y_uncertainty"], eval_dict["u_probs"])
        display_roc_curve(eval_dict["y_tumour"], eval_dict["t_probs"])
        display_roc_curve(eval_dict["y_node"], eval_dict["n_probs"])
        display_roc_curve(eval_dict["y_mets"], eval_dict["m_probs"])
