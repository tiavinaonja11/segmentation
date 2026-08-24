"""
Utilitaires partagés pour charger le modèle et faire des prédictions —
utilisés par inference.py (CLI), app.py (interface Streamlit) et
api.py (service REST).
"""

import io
import tempfile
from pathlib import Path

import numpy as np
import torch

from data.dataset import load_ai4boundaries_image
from models.unet import UNet


def load_model(checkpoint_path="checkpoints/best_model.pt", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]

    model = UNet(
        n_bands=cfg["data"]["n_bands"],
        n_classes=cfg["data"]["n_classes"],
        base_channels=cfg["model"]["base_channels"],
        depth=cfg["model"]["depth"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, cfg, checkpoint, device


def load_input_image(file_bytes, filename, n_bands):
    """Charge une image .nc (30 canaux Sentinel-2, normalisée comme à
    l'entraînement) ou .npy (déjà normalisée), à partir de bytes bruts."""

    suffix = Path(filename).suffix.lower()

    if suffix == ".npy":
        image = np.load(io.BytesIO(file_bytes)).astype(np.float32)
        image = image / max(image.max(), 1e-6)
        return image

    if suffix == ".nc":
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            return load_ai4boundaries_image(tmp_path, n_bands=n_bands)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    raise ValueError(f"Format non supporté : {suffix} (attendu : .nc ou .npy)")


def predict(model, device, image, threshold=0.5):
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask = (probs > threshold).astype(np.uint8)
    return probs, mask
