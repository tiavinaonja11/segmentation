"""
Fonctions de visualisation : image / masque réel / masque prédit.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def _to_rgb(image_tensor):
    """Convertit un tenseur multi-bandes (C,H,W) en image RGB affichable."""
    img = image_tensor.cpu().numpy()
    if img.shape[0] >= 3:
        rgb = img[:3]
    else:
        rgb = np.repeat(img[:1], 3, axis=0)
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    return rgb


def save_prediction_grid(images, masks, preds, out_path, n_samples=4):
    """Sauvegarde une grille (image | masque réel | prédiction) pour n_samples exemples."""
    n_samples = min(n_samples, images.shape[0])
    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_samples):
        rgb = _to_rgb(images[i])
        mask = masks[i, 0].cpu().numpy()
        pred = torch.sigmoid(preds[i, 0]).cpu().numpy()

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title("Image (RGB)")
        axes[i, 1].imshow(mask, cmap="Greens")
        axes[i, 1].set_title("Masque réel")
        axes[i, 2].imshow(pred, cmap="Greens", vmin=0, vmax=1)
        axes[i, 2].set_title("Prédiction")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_training_curves(history, out_path):
    """history: dict avec listes 'train_loss', 'val_loss', 'val_iou', 'val_dice'"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train loss")
    axes[0].plot(history["val_loss"], label="Val loss")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Perte")
    axes[0].set_title("Courbe de perte")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["val_iou"], label="Val IoU", color="green")
    axes[1].plot(history["val_dice"], label="Val Dice", color="orange")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Métriques de validation")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
