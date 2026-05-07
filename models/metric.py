import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, recall_score 
from sklearn.metrics import precision_score, confusion_matrix, ConfusionMatrixDisplay


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def compute_scores(preds, label_set):
    scores = {'F1_MACRO': f1_score(label_set, preds, average="macro"),
              'F1_MICRO': f1_score(label_set, preds, average="micro"),
              'RECALL_MACRO': recall_score(label_set, preds, average="macro"),
              'RECALL_MICRO': recall_score(label_set, preds, average="micro"),
              'PRECISION_MACRO': precision_score(label_set, preds, average="macro"),
              'PRECISION_MICRO': precision_score(label_set, preds, average="micro")}

    return scores


def display_confusion_matrix(label_set, preds):
    cm = confusion_matrix(label_set, preds)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.show()


def display_roc_curve(label_set, probs):
    fpr, tpr, _ = roc_curve(label_set, probs)

    roc_auc = auc(fpr, tpr)
    print(f"Val AUC: {roc_auc}")

    plt.plot(fpr, tpr, marker='.', label='AUC = %0.2f' % roc_auc)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc = 'lower right')
    plt.show()
