"""
Fonctions de perte pour la segmentation binaire / multi-classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """Combinaison BCE + Dice, robuste pour les masques déséquilibrés
    (peu de pixels de bordure vs beaucoup de pixels de fond)."""

    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


class FocalLoss(nn.Module):
    """Utile quand les parcelles occupent une faible portion de l'image
    (fort déséquilibre de classes)."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


def get_loss_fn(name: str):
    name = name.lower()
    if name == "dice":
        return DiceLoss()
    elif name == "bce_dice":
        return BCEDiceLoss()
    elif name == "focal":
        return FocalLoss()
    else:
        raise ValueError(f"Loss inconnue : {name}")


# ---------------------------------------------------------------
# Pour la segmentation MULTI-CLASSES (ex: types de cultures) :
# remplacez get_loss_fn par nn.CrossEntropyLoss() et adaptez les
# masques pour qu'ils contiennent des indices de classe (int64)
# plutôt que des valeurs binaires 0/1.
# ---------------------------------------------------------------
