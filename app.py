"""
Interface web pour tester le modèle de segmentation de parcelles agricoles
sur une image fournie par l'utilisateur.

Usage :
    streamlit run app.py
"""

import io
import os
from pathlib import Path

import numpy as np
import streamlit as st

from serving import load_input_image, load_model as _load_model, predict as _predict

LOCAL_CHECKPOINT_PATH = "checkpoints/best_model.pt"

# Sur Hugging Face Spaces, le checkpoint n'est pas embarqué dans le Space :
# il est téléchargé depuis le repo modèle au démarrage.
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "tiavinaonja/star")
HF_MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME", "best_model.pt")


@st.cache_resource
def load_model():
    checkpoint_path = LOCAL_CHECKPOINT_PATH
    if not Path(checkpoint_path).exists():
        from huggingface_hub import hf_hub_download
        checkpoint_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILENAME)
    return _load_model(checkpoint_path)


def to_rgb_preview(image):
    if image.shape[0] == 30:
        # B2(0-5) B3(6-11) B4(12-17) B8(18-23) NDVI(24-29)
        # Vraies couleurs (R=B4, G=B3, B=B2) sur la dernière date.
        rgb = image[[17, 11, 5]]
    elif image.shape[0] >= 3:
        rgb = image[:3]
    else:
        rgb = np.repeat(image[:1], 3, axis=0)

    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    return rgb


def colorize(array, cmap_low, cmap_high):
    """Convertit un tableau 2D [0,1] en image RGB en interpolant entre deux couleurs."""
    array = np.clip(array, 0, 1)[..., None]
    low = np.array(cmap_low, dtype=np.float32)
    high = np.array(cmap_high, dtype=np.float32)
    return (low + array * (high - low)).astype(np.float32)


st.set_page_config(page_title="Segmentation de parcelles agricoles", layout="wide")
st.title("🌾 Segmentation de parcelles agricoles")
st.caption(
    "Charge une image Sentinel-2 multi-temporelle (.nc, 30 canaux — B2/B3/B4/B8/NDVI × 6 dates) "
    "ou un tableau .npy déjà au bon format, pour tester le modèle U-Net entraîné."
)

try:
    model, cfg, checkpoint, device = load_model()
except Exception as e:
    st.error(f"Impossible de charger le modèle : {e}")
    st.stop()

with st.sidebar:
    st.subheader("Modèle")
    st.write(f"Époque : **{checkpoint['epoch']}**")
    st.write(f"IoU validation : **{checkpoint['val_iou']:.4f}**")
    st.write(f"Canaux attendus : **{cfg['data']['n_bands']}**")
    st.divider()
    threshold = st.slider("Seuil de segmentation", 0.0, 1.0, 0.5, 0.01)

uploaded_file = st.file_uploader(
    "Image d'entrée (.nc ou .npy)",
    type=["nc", "npy"],
)

if uploaded_file is not None:
    try:
        image = load_input_image(uploaded_file.getvalue(), uploaded_file.name, n_bands=cfg["data"]["n_bands"])
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'image : {e}")
        st.stop()

    if image.shape[0] != cfg["data"]["n_bands"]:
        st.error(
            f"Nombre de canaux incorrect : {image.shape[0]} au lieu de "
            f"{cfg['data']['n_bands']} attendus par le modèle."
        )
        st.stop()

    probs, pred_mask = _predict(model, device, image, threshold=threshold)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Image d'entrée")
        st.image(to_rgb_preview(image), use_container_width=True)
    with col2:
        st.subheader("Carte de probabilité")
        st.image(colorize(probs, (1, 1, 1), (0.0, 0.4, 0.0)), use_container_width=True)
    with col3:
        st.subheader(f"Masque prédit (seuil={threshold:.2f})")
        st.image(colorize(pred_mask.astype(np.float32), (1, 1, 1), (0.0, 0.4, 0.0)), use_container_width=True)

    field_pct = 100 * pred_mask.mean()
    st.metric("Surface classée « parcelle »", f"{field_pct:.1f}%")

    mask_bytes = io.BytesIO()
    np.save(mask_bytes, pred_mask)
    st.download_button(
        "Télécharger le masque (.npy)",
        data=mask_bytes.getvalue(),
        file_name="predicted_mask.npy",
    )
else:
    st.info("Charge un fichier .nc ou .npy pour lancer la prédiction.")
