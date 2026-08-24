"""
Génère un jeu de données SYNTHÉTIQUE (images multi-bandes + masques de parcelles)
pour tester immédiatement tout le pipeline sans attendre le téléchargement du
vrai dataset AI4Boundaries.

Les "parcelles" sont simulées par des polygones (rectangles/quadrilatères)
répartis aléatoirement, avec un bruit texturé simulant la variabilité spectrale
des cultures.

Usage:
    python data/generate_synthetic_data.py --out data/synthetic --n_samples 200
"""

import os
import argparse

import numpy as np
from PIL import Image, ImageDraw


def make_field_mask(size, n_fields_range=(3, 8)):
    """Crée un masque binaire de parcelles + une image RGB+NIR simulée."""
    h, w = size, size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    n_fields = np.random.randint(*n_fields_range)

    # découpe l'image en une grille irrégulière façon "voronoi grossier"
    for _ in range(n_fields):
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        rw, rh = np.random.randint(w // 6, w // 2), np.random.randint(h // 6, h // 2)
        angle = np.random.uniform(0, 30)

        x0, y0 = cx - rw // 2, cy - rh // 2
        x1, y1 = cx + rw // 2, cy + rh // 2

        # bordure noire (non-parcelle) autour du polygone plein (valeur=1)
        draw.rectangle([x0, y0, x1, y1], fill=1)

    mask_arr = np.array(mask, dtype=np.uint8)

    # Ajout de bordures nettes entre parcelles (érosion simple pour créer un contour à 0)
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(mask_arr, iterations=1).astype(np.uint8)
    mask_arr = eroded  # bordures = 0, intérieur de parcelle = 1

    return mask_arr


def make_image(mask_arr, n_bands=4):
    """Simule une image multi-bandes cohérente avec le masque (texture par parcelle)."""
    h, w = mask_arr.shape
    image = np.zeros((n_bands, h, w), dtype=np.float32)

    # fond (sol nu / non-parcelle)
    base_bg = np.random.uniform(0.2, 0.4, size=(n_bands, 1, 1))
    image[:] = base_bg + np.random.normal(0, 0.02, size=(n_bands, h, w))

    # simule des parcelles avec des couleurs/textures différentes via labels connectés
    from scipy.ndimage import label
    labeled, n_comp = label(mask_arr)
    for i in range(1, n_comp + 1):
        comp_mask = labeled == i
        color = np.random.uniform(0.3, 0.9, size=(n_bands, 1, 1))
        noise = np.random.normal(0, 0.03, size=(n_bands, h, w))
        image[:, comp_mask] = (color + noise)[:, comp_mask]

    image = np.clip(image, 0, 1)
    return image


def generate_dataset(out_dir, n_samples=200, size=256, n_bands=4, seed=42):
    np.random.seed(seed)

    img_dir = os.path.join(out_dir, "images")
    mask_dir = os.path.join(out_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for i in range(n_samples):
        mask_arr = make_field_mask(size)
        image_arr = make_image(mask_arr, n_bands)

        fname = f"sample_{i:04d}.npy"
        np.save(os.path.join(img_dir, fname), image_arr)
        np.save(os.path.join(mask_dir, fname), mask_arr[np.newaxis, ...])

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_samples} échantillons générés")

    print(f"\n✅ Dataset synthétique généré dans : {out_dir}")
    print(f"   {n_samples} échantillons, taille {size}x{size}, {n_bands} bandes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère un dataset synthétique de parcelles agricoles")
    parser.add_argument("--out", type=str, default="data/synthetic", help="Dossier de sortie")
    parser.add_argument("--n_samples", type=int, default=200, help="Nombre d'échantillons")
    parser.add_argument("--size", type=int, default=256, help="Taille des patches (pixels)")
    parser.add_argument("--n_bands", type=int, default=4, help="Nombre de bandes spectrales")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(args.out, args.n_samples, args.size, args.n_bands, args.seed)
