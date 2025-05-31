import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import wandb

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    distances = []
    labels = []
    all_scores = []
    all_labels = []
    for anchor, positive, negative in dataloader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)
        d_ap = F.pairwise_distance(anchor_out, positive_out).cpu().numpy()
        d_an = F.pairwise_distance(anchor_out, negative_out).cpu().numpy()
        distances.extend(list(d_ap) + list(d_an))
        labels.extend([1] * len(d_ap) + [0] * len(d_an))
        all_scores.extend(list(-d_ap) + list(-d_an))
        all_labels.extend([1] * len(d_ap) + [0] * len(d_an))

    distances = np.array(distances)
    labels = np.array(labels)
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    thresholds = np.linspace(distances.min(), distances.max(), 100)
    best_thresh = 0
    best_acc = 0
    accuracies = []

    for thresh in thresholds:
        preds = distances <= thresh
        acc = np.mean(preds == labels)
        accuracies.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)

    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

    wandb.log({"best_accuracy": best_acc, "roc_plot": wandb.Image('roc_curve.png') })
    #print(f"Validation - Best Threshold: {best_thresh:.4f}, Accuracy: {best_acc * 100:.2f}%")
    fpr, tpr, _ = roc_curve(labels, -distances)
    wandb.log({ "roc_curve": wandb.plot.roc_curve(labels, [(-distances, "Model")]) })
    return best_thresh, best_acc

