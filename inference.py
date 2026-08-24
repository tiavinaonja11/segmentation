"""
Applique un modèle entraîné à une nouvelle image satellite pour prédire les
limites de parcelles agricoles.

Usage:
    python inference.py --checkpoint checkpoints/best_model.pt --image path/to/image.tif --out outputs/prediction.png
"""

import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from data.dataset import load_ai4boundaries_image
from serving import load_model, predict

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def load_image(path, n_bands):
    if path.endswith(".nc"):
        # Même chargement + normalisation que pendant l'entraînement
        # (voir data/dataset.py::load_ai4boundaries_image), pour que
        # le modèle reçoive une entrée dans la même distribution.
        return load_ai4boundaries_image(path, n_bands=n_bands), True
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32), False
    if HAS_RASTERIO:
        with rasterio.open(path) as src:
            return src.read().astype(np.float32), False
    raise RuntimeError("Installez rasterio pour lire les fichiers .tif : pip install rasterio")


def main(args):
    model, cfg, checkpoint, device = load_model(args.checkpoint)

    print(f"Modèle chargé (époque {checkpoint['epoch']}, IoU val={checkpoint['val_iou']:.4f})")

    image, already_normalized = load_image(args.image, n_bands=cfg["data"]["n_bands"])
    if not already_normalized:
        image = image / max(image.max(), 1e-6)

    probs, pred_mask = predict(model, device, image, threshold=args.threshold)

    # visualisation
    if image.shape[0] == 30:
        # Ordre des canaux : B2(0-5) B3(6-11) B4(12-17) B8(18-23) NDVI(24-29)
        # Composition vraies couleurs (R=B4, G=B3, B=B2) sur la dernière date.
        rgb = image[[17, 11, 5]]
    elif image.shape[0] >= 3:
        rgb = image[:3]
    else:
        rgb = np.repeat(image[:1], 3, axis=0)
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Image d'entrée")
    axes[1].imshow(probs, cmap="Greens", vmin=0, vmax=1)
    axes[1].set_title("Carte de probabilité")
    axes[2].imshow(pred_mask, cmap="Greens")
    axes[2].set_title(f"Masque prédit (seuil={args.threshold})")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"✅ Résultat sauvegardé dans : {args.out}")

    # sauvegarde également le masque brut en .npy pour usage ultérieur (SIG, etc.)
    mask_out = args.out.rsplit(".", 1)[0] + "_mask.npy"
    np.save(mask_out, pred_mask)
    print(f"   Masque binaire sauvegardé dans : {mask_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inférence de segmentation de parcelles agricoles")
    parser.add_argument("--checkpoint", type=str, required=True, help="Chemin vers le modèle entraîné (.pt)")
    parser.add_argument("--image", type=str, required=True, help="Image d'entrée (.tif ou .npy)")
    parser.add_argument("--out", type=str, default="outputs/prediction.png")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    main(args)
