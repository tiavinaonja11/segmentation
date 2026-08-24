"""
Récupère une image Sentinel-2 multi-temporelle (B2, B3, B4, B8, NDVI × 6 dates)
pour une zone d'intérêt donnée, via Google Earth Engine, et la sauvegarde au
format .nc — même structure que les données AI4Boundaries utilisées à
l'entraînement (voir data/dataset.py). Le fichier produit peut ensuite être
utilisé directement avec inference.py.

Prérequis :
    pip install earthengine-api
    earthengine authenticate   # (une seule fois, ouvre un navigateur)

Usage :
    python data/fetch_gee_aoi.py \
        --lat -19.855240 --lon 47.009548 \
        --days 150 \
        --out data/custom/antananarivo.nc
"""

import argparse
import io
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

try:
    import ee
except ImportError as e:
    raise SystemExit(
        "Le paquet 'earthengine-api' est requis : pip install earthengine-api"
    ) from e

import urllib.request

BANDS = ["B2", "B3", "B4", "B8"]


def utm_epsg(lat, lon):
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:{32700 + zone if lat < 0 else 32600 + zone}"


def mask_s2_clouds(image):
    qa = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = (
        qa.bitwiseAnd(cloud_bit).eq(0)
        .And(qa.bitwiseAnd(cirrus_bit).eq(0))
    )
    return image.updateMask(mask)


def date_windows(end_date, days, n_windows):
    start_date = end_date - timedelta(days=days)
    window_len = timedelta(days=days / n_windows)
    windows = []
    for i in range(n_windows):
        w_start = start_date + i * window_len
        w_end = w_start + window_len
        windows.append((w_start, w_end))
    return windows


def fetch_window_arrays(region, crs, patch_px, w_start, w_end):
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(w_start.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d"))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
        .map(mask_s2_clouds)
    )

    if collection.size().getInfo() == 0:
        raise RuntimeError(
            f"Aucune image Sentinel-2 disponible entre {w_start.date()} "
            f"et {w_end.date()} pour cette zone (essayez d'élargir --days)."
        )

    composite = collection.select(BANDS).median()
    ndvi = composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
    image = composite.addBands(ndvi).toFloat()

    url = image.getDownloadURL({
        "region": region,
        "dimensions": f"{patch_px}x{patch_px}",
        "crs": crs,
        "format": "NPY",
    })

    with urllib.request.urlopen(url) as response:
        raw = response.read()

    arr = np.load(io.BytesIO(raw))  # structured array (H, W) with named fields

    return {name: arr[name].astype(np.float32) for name in BANDS + ["NDVI"]}


def main(args):
    ee.Initialize(project=args.ee_project) if args.ee_project else ee.Initialize()

    end_date = (
        datetime.strptime(args.end_date, "%Y-%m-%d")
        if args.end_date else datetime.utcnow()
    )

    crs = utm_epsg(args.lat, args.lon)
    half_size_m = args.patch_px * args.scale / 2
    region = (
        ee.Geometry.Point([args.lon, args.lat])
        .buffer(half_size_m, 1)
        .bounds()
    )

    windows = date_windows(end_date, args.days, args.n_dates)

    per_variable = {name: [] for name in BANDS + ["NDVI"]}
    for i, (w_start, w_end) in enumerate(windows):
        print(f"[{i + 1}/{len(windows)}] {w_start.date()} -> {w_end.date()} ...")
        arrays = fetch_window_arrays(region, crs, args.patch_px, w_start, w_end)
        for name, arr in arrays.items():
            per_variable[name].append(arr)

    data_vars = {
        name: (("time", "y", "x"), np.stack(stack, axis=0))
        for name, stack in per_variable.items()
    }
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": [w[0] for w in windows]},
    )

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ds.to_netcdf(args.out)
    print(f"✅ Image sauvegardée dans : {args.out}")
    print("   Lancez maintenant :")
    print(
        f"   python inference.py --checkpoint checkpoints/best_model.pt "
        f"--image {args.out} --out outputs/prediction.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Télécharge une zone Sentinel-2 (Google Earth Engine) au format AI4Boundaries"
    )
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--days", type=int, default=150, help="Fenêtre temporelle totale, en jours")
    parser.add_argument("--n-dates", type=int, default=6, help="Nombre de composites (doit correspondre au modèle)")
    parser.add_argument("--end-date", type=str, default=None, help="Date de fin (AAAA-MM-JJ), défaut = aujourd'hui")
    parser.add_argument("--patch-px", type=int, default=256, help="Taille du patch en pixels")
    parser.add_argument("--scale", type=float, default=10.0, help="Résolution en mètres/pixel")
    parser.add_argument("--ee-project", type=str, default=None, help="ID du projet Google Cloud lié à Earth Engine")
    parser.add_argument("--out", type=str, default="data/custom/aoi.nc")
    args = parser.parse_args()

    main(args)
