import torch
import torch.nn.functional as F
import numpy as np


# Below functions adapted from: https://github.com/ieshanvaidya/UDA
def get_tsa_threshold(schedule, t, T, K):
    """
    schedule: log, linear, exp
    t: training step
    T: total steps
    K: number of categories
    """
    step_ratio = torch.tensor(t / T)
    if schedule == 'log':
        alpha = 1 - torch.exp(-5 * step_ratio)
    elif schedule == 'linear':
        alpha = step_ratio
    elif schedule == 'exp':
        alpha = torch.exp(5 * (step_ratio - 1))
    else:
        raise ValueError('Invalid schedule')

    threshold = alpha * (1 - 1 / K) + 1 / K
    return threshold


def compute_uda_loss(device, model, supervised_batch, unsupervised_batch, supervised_criterion, unsupervised_criterion,
                 training_step, args, total_steps):
    model.train()

    supervised_encoding, supervised_labels = supervised_batch
    ori_encoding, aug_encoding = unsupervised_batch

    input_ids = torch.cat((supervised_encoding['input_ids'], aug_encoding['input_ids']), dim=0).to(device)
    attention_mask = torch.cat((supervised_encoding['attention_mask'], aug_encoding['attention_mask']), dim=0).to(device)

    supervised_labels = supervised_labels.to(device)

    original_input_ids = ori_encoding['input_ids'].to(device)
    original_attention_mask = ori_encoding['attention_mask'].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask)


    # Supervised Loss
    # ---------------
    num_supervised = supervised_labels.shape[0]
    supervised_loss = supervised_criterion(logits[:num_supervised], supervised_labels)
    if args.tsa:
        threshold = get_tsa_threshold(args.schedule, training_step, total_steps, logits.shape[-1])

        # Probabilities: exp(-loss)
        larger_than_threshold = torch.exp(-supervised_loss) > threshold

        # Mask those below threshold
        supervised_mask = torch.ones_like(supervised_labels, dtype=torch.float32) * (
                    1 - larger_than_threshold.type(torch.float32))

        # Recompute normalized loss
        supervised_loss = torch.sum(supervised_loss * supervised_mask, dim=-1) / torch.max(
            torch.sum(supervised_mask, dim=-1), torch.tensor(1.).to(device))
        

    else:
        supervised_loss = torch.mean(supervised_loss)

    # Unsupervised Loss
    # -----------------
    with torch.no_grad():

        original_logits = model(input_ids=original_input_ids,
                                attention_mask=original_attention_mask)
        # KL divergence target
        original_probs = torch.softmax(original_logits, dim=-1)
        # Confidence based masking
        if args.uda_confidence_threshold != -1:
            unsupervised_mask = torch.max(original_probs, dim=-1)[0] > args.uda_confidence_threshold
            unsupervised_mask = unsupervised_mask.type(torch.float32)
        else:
            unsupervised_mask = torch.ones(len(logits) - num_supervised, dtype=torch.float32)

        unsupervised_mask = unsupervised_mask.to(device)

    uda_softmax_temp = args.uda_softmax_temperature if args.uda_softmax_temperature > 0 else 1
    augmented_log_probs = torch.log_softmax(logits[num_supervised:] / uda_softmax_temp, dim=-1)

    # Using SanghunYun's version (https://github.com/SanghunYun/UDA_pytorch)
    unsupervised_loss = torch.sum(unsupervised_criterion(augmented_log_probs, original_probs), dim=-1)
    unsupervised_loss = torch.sum(unsupervised_loss * unsupervised_mask, dim=-1) / torch.max(
        torch.sum(unsupervised_mask, dim=-1), torch.tensor(1.).to(device))

    final_loss = supervised_loss + args.uda_coefficient * unsupervised_loss
    return final_loss, supervised_loss, unsupervised_loss

def compute_supervised_loss(device, model, supervised_batch, supervised_criterion,
                 training_step, args, total_steps):
    model.train()

    supervised_encoding, supervised_labels = supervised_batch

    input_ids = supervised_encoding['input_ids'].to(device)
    attention_mask = supervised_encoding['attention_mask'].to(device)

    supervised_labels = supervised_labels.to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask)

    supervised_loss = supervised_criterion(logits, supervised_labels)
    if args.tsa:
        threshold = get_tsa_threshold(args.schedule, training_step, total_steps, logits.shape[-1])

        # Probabilities: exp(-loss)
        larger_than_threshold = torch.exp(-supervised_loss) > threshold

        # Mask those below threshold
        supervised_mask = torch.ones_like(supervised_labels, dtype=torch.float32) * (
                    1 - larger_than_threshold.type(torch.float32))

        # Recompute normalized loss
        supervised_loss = torch.sum(supervised_loss * supervised_mask, dim=-1) / torch.max(
            torch.sum(supervised_mask, dim=-1), torch.tensor(1.).to(device))
    else:
        supervised_loss = torch.mean(supervised_loss)

    return supervised_loss

def evaluate(device, model, dataloader):
    model.eval()

    correct = 0
    total = 0
    for batch in dataloader:
        encoding, labels = batch
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        labels = labels.to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        _, predictions = logits.max(1)

        correct += (predictions == labels).float().sum().item()
        total += input_ids.shape[0]

    return correct / total



def eval_model(device, model, dataloader, supervised_criterion):

        model = model.eval()
        probs = []
        predictions = []
        y_test_labels = []
        losses = []

        with torch.no_grad():
            for batch in dataloader:
                encoding, labels = batch
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)
                labels = labels.to(device)
                logits = model(input_ids=input_ids,
                                        attention_mask=attention_mask)
                _, preds = torch.max(logits, dim=1)
                predictions.append(preds)
                prob = F.softmax(logits, dim=1)
                probs.append(prob)
                y_test_labels.append(labels)
                loss = supervised_criterion(logits, labels)
                losses.append(loss.item())
        y_test_labels = torch.cat(y_test_labels).cpu().data.numpy()
        predictions = torch.cat(predictions).squeeze().cpu().data.numpy()
        probs = torch.cat(probs).cpu().data.numpy()

        return {
                "y_labels": y_test_labels, 
                "preds": predictions, 
                "probs": probs,
                "loss": np.mean(losses)
                }
