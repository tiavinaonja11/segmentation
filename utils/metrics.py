"""
Métriques de segmentation : IoU, Dice, précision, rappel.
"""

import torch


@torch.no_grad()
def compute_metrics(logits, targets, threshold=0.5, eps=1e-7):
    """
    Args:
        logits: sortie brute du modèle (avant sigmoïde), shape (B, 1, H, W)
        targets: masques binaires (0/1), shape (B, 1, H, W)
    Returns:
        dict avec iou, dice, precision, recall
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    tp = (preds_flat * targets_flat).sum()
    fp = (preds_flat * (1 - targets_flat)).sum()
    fn = ((1 - preds_flat) * targets_flat).sum()

    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    return {
        "iou": iou.item(),
        "dice": dice.item(),
        "precision": precision.item(),
        "recall": recall.item(),
    }


class AverageMeter:
    """Suit la moyenne d'une métrique sur les batchs d'une époque."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)
