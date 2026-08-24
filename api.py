"""
API REST pour la segmentation de parcelles agricoles.

Usage :
    uvicorn api:app --host 0.0.0.0 --port 8000

Exemple d'appel :
    curl -F "file=@data/AI4Boundaries/test/images/AT_10038_S2_10m_256.nc" \
         "http://localhost:8000/predict?threshold=0.5"
"""

import base64
import io

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

from serving import load_input_image, load_model, predict

app = FastAPI(
    title="Agri Field Segmentation API",
    description="Segmentation de parcelles agricoles à partir d'imagerie Sentinel-2 multi-temporelle (U-Net).",
    version="1.0.0",
)

# Permet d'appeler l'API directement depuis un navigateur (autre origine/port).
# À restreindre à ton propre domaine en production (allow_origins=[...]).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {}


@app.on_event("startup")
def _startup():
    model, cfg, checkpoint, device = load_model()
    _state["model"] = model
    _state["cfg"] = cfg
    _state["checkpoint"] = checkpoint
    _state["device"] = device


class ModelInfo(BaseModel):
    epoch: int
    val_iou: float
    n_bands: int
    n_classes: int


class PredictResponse(BaseModel):
    threshold: float
    field_percentage: float
    mask_shape: list[int]
    mask_png_base64: str
    probability_png_base64: str


def _array_to_png_base64(array_uint8):
    buf = io.BytesIO()
    Image.fromarray(array_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info", response_model=ModelInfo)
def info():
    checkpoint = _state["checkpoint"]
    cfg = _state["cfg"]
    return ModelInfo(
        epoch=checkpoint["epoch"],
        val_iou=checkpoint["val_iou"],
        n_bands=cfg["data"]["n_bands"],
        n_classes=cfg["data"]["n_classes"],
    )


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(..., description="Image .nc (30 canaux) ou .npy déjà normalisée"),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    format: str = Query("json", pattern="^(json|mask_png)$"),
):
    cfg = _state["cfg"]
    n_bands = cfg["data"]["n_bands"]

    file_bytes = await file.read()
    try:
        image = load_input_image(file_bytes, file.filename, n_bands=n_bands)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if image.shape[0] != n_bands:
        raise HTTPException(
            status_code=400,
            detail=f"Nombre de canaux incorrect : {image.shape[0]} au lieu de {n_bands} attendus.",
        )

    probs, mask = predict(_state["model"], _state["device"], image, threshold=threshold)

    if format == "mask_png":
        png_bytes = io.BytesIO()
        Image.fromarray((mask * 255).astype(np.uint8)).save(png_bytes, format="PNG")
        return Response(content=png_bytes.getvalue(), media_type="image/png")

    return PredictResponse(
        threshold=threshold,
        field_percentage=float(100 * mask.mean()),
        mask_shape=list(mask.shape),
        mask_png_base64=_array_to_png_base64((mask * 255).astype(np.uint8)),
        probability_png_base64=_array_to_png_base64((probs * 255).astype(np.uint8)),
    )
