"""Metrics and evaluation utilities"""

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import numpy as np


def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    criterion = nn.BCELoss()
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            running_loss += criterion(outputs, labels).item() * images.size(0)
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(outputs.cpu().numpy().flatten())

    labels, probs = np.array(all_labels), np.array(all_probs)
    preds = (probs > 0.5).astype(float)

    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0

    return {
        'loss': running_loss / len(labels),
        'accuracy': (tp + tn) / len(labels),
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'auc': auc,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'labels': labels,
        'probs': probs
    }