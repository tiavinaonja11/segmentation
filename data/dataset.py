"""
Dataset AI4Boundaries pour segmentation de parcelles agricoles.

Structure attendue :

data/AI4Boundaries/
    train/
        images/*.nc
        masks/*.tif
    val/
        images/*.nc
        masks/*.tif
    test/
        images/*.nc
        masks/*.tif

Chaque fichier .nc contient :

    B2   : 6 dates
    B3   : 6 dates
    B4   : 6 dates
    B8   : 6 dates
    NDVI : 6 dates

=> 5 variables × 6 dates = 30 canaux.

Le dataset :
    - associe uniquement les images .nc aux masques .tif correspondants
    - ignore les masques orphelins
    - normalise les bandes Sentinel-2
    - conserve NDVI dans sa plage naturelle
    - remplace NaN / Inf
    - binarise les masques
    - applique les augmentations uniquement au train
"""

from pathlib import Path

import numpy as np
import torch
import xarray as xr
import rasterio

from torch.utils.data import Dataset, DataLoader

import albumentations as A


# ============================================================
# VARIABLES UTILISÉES
# ============================================================

VARIABLES = [
    "B2",
    "B3",
    "B4",
    "B8",
    "NDVI",
]


# ============================================================
# AUGMENTATION TRAIN
# ============================================================

def get_train_augmentation(cfg):
    """
    Augmentations géométriques uniquement.

    Important :
    On évite ici les transformations de luminosité/contraste
    car les données sont multispectrales et contiennent
    des bandes physiques ainsi que le NDVI.
    """

    aug_cfg = cfg.get("augmentation", {})

    return A.Compose([
        A.HorizontalFlip(
            p=aug_cfg.get("horizontal_flip", 0.5)
        ),

        A.VerticalFlip(
            p=aug_cfg.get("vertical_flip", 0.5)
        ),

        A.RandomRotate90(
            p=aug_cfg.get("rotate90", 0.5)
        ),
    ])


# ============================================================
# CHARGEMENT + NORMALISATION D'UNE IMAGE .nc
#
# Partagé entre le Dataset (entraînement/validation) et
# inference.py, pour garantir que les images inférées
# reçoivent exactement la même normalisation que pendant
# l'entraînement.
# ============================================================

def load_ai4boundaries_image(path, n_bands=30):

    try:

        ds = xr.open_dataset(path)

    except Exception as e:

        raise RuntimeError(
            f"Impossible d'ouvrir : {path}\n"
            f"Erreur : {e}"
        )

    channels = []

    try:

        for variable in VARIABLES:

            # --------------------------------------------
            # Vérifier variable
            # --------------------------------------------

            if variable not in ds:

                raise RuntimeError(
                    f"Variable '{variable}' absente "
                    f"de {path}"
                )

            data = ds[variable].values

            # --------------------------------------------
            # Vérifier dimension
            # --------------------------------------------

            if data.ndim != 3:

                raise RuntimeError(
                    f"{variable} possède une forme "
                    f"inattendue : {data.shape}"
                )

            # --------------------------------------------
            # Vérifier nombre de dates
            # --------------------------------------------

            if data.shape[0] != 6:

                raise RuntimeError(
                    f"{variable} devrait contenir "
                    f"6 dates mais possède "
                    f"{data.shape[0]}"
                )

            # --------------------------------------------
            # Float32
            # --------------------------------------------

            data = data.astype(
                np.float32
            )

            channels.append(data)

    finally:

        ds.close()

    # ====================================================
    # 5 × 6 × H × W
    #
    # ->
    #
    # 30 × H × W
    # ====================================================

    image = np.concatenate(
        channels,
        axis=0
    )

    # ====================================================
    # Vérification du nombre de bandes
    # ====================================================

    if image.shape[0] != n_bands:

        raise RuntimeError(
            f"Nombre de canaux incorrect pour {path} : "
            f"{image.shape[0]} au lieu de "
            f"{n_bands}"
        )

    # ====================================================
    # Nettoyage NaN / Inf
    # ====================================================

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # ====================================================
    # NORMALISATION
    # ====================================================

    normalized = np.zeros_like(
        image,
        dtype=np.float32
    )

    # ----------------------------------------------------
    # Chaque variable possède 6 dates.
    #
    # 0-5   = B2
    # 6-11  = B3
    # 12-17 = B4
    # 18-23 = B8
    # 24-29 = NDVI
    # ----------------------------------------------------

    for i, variable in enumerate(VARIABLES):

        start = i * 6
        end = start + 6

        block = image[start:end]

        if variable == "NDVI":

            # --------------------------------------------
            # NDVI naturel :
            # environ [-1, 1]
            #
            # On le convertit vers [0, 1]
            # --------------------------------------------

            block = np.clip(
                block,
                -1.0,
                1.0
            )

            block = (
                block + 1.0
            ) / 2.0

        else:

            # --------------------------------------------
            # Sentinel-2 reflectance stockée
            # sur une échelle pouvant aller jusqu'à
            # plusieurs milliers.
            #
            # Normalisation robuste vers [0,1].
            # --------------------------------------------

            block = np.clip(
                block,
                0.0,
                10000.0
            )

            block = (
                block / 10000.0
            )

        normalized[start:end] = block

    # ----------------------------------------------------
    # Sécurité finale
    # ----------------------------------------------------

    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )

    return normalized.astype(
        np.float32
    )


# ============================================================
# DATASET
# ============================================================

class FieldSegmentationDataset(Dataset):

    def __init__(
        self,
        data_dir,
        split="train",
        transform=None,
        n_bands=30,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.n_bands = n_bands

        self.image_dir = (
            self.data_dir /
            split /
            "images"
        )

        self.mask_dir = (
            self.data_dir /
            split /
            "masks"
        )

        # ----------------------------------------------------
        # Vérification des dossiers
        # ----------------------------------------------------

        if not self.image_dir.exists():
            raise RuntimeError(
                f"Dossier images introuvable : "
                f"{self.image_dir}"
            )

        if not self.mask_dir.exists():
            raise RuntimeError(
                f"Dossier masks introuvable : "
                f"{self.mask_dir}"
            )

        # ----------------------------------------------------
        # Récupération des images
        # ----------------------------------------------------

        image_files = sorted(
            self.image_dir.glob("*.nc")
        )

        if not image_files:
            raise RuntimeError(
                f"Aucune image .nc trouvée dans "
                f"{self.image_dir}"
            )

        # ----------------------------------------------------
        # Construire uniquement les vraies paires
        #
        # image :
        # AT_10039_S2_10m_256.nc
        #
        # mask :
        # AT_10039_S2_10m_256.tif
        # ----------------------------------------------------

        self.files = []

        missing_masks = []

        for image_path in image_files:

            mask_path = (
                self.mask_dir /
                f"{image_path.stem}.tif"
            )

            if mask_path.exists():

                self.files.append(
                    image_path
                )

            else:

                missing_masks.append(
                    image_path.name
                )

        # ----------------------------------------------------
        # Vérification
        # ----------------------------------------------------

        if not self.files:
            raise RuntimeError(
                f"Aucune paire image/mask trouvée "
                f"dans {self.data_dir}/{split}"
            )

        print(
            f"[{split}] "
            f"{len(self.files)} paires image/mask trouvées"
        )

        if missing_masks:
            print(
                f"[{split}] "
                f"{len(missing_masks)} images sans masque ignorées"
            )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):
        return len(self.files)

    # ========================================================
    # CHARGEMENT IMAGE
    # ========================================================

    def _load_image(self, path):

        return load_ai4boundaries_image(
            path,
            n_bands=self.n_bands,
        )

    # ========================================================
    # CHARGEMENT MASQUE
    # ========================================================

    def _load_mask(self, image_path):

        mask_name = (
            image_path.stem +
            ".tif"
        )

        mask_path = (
            self.mask_dir /
            mask_name
        )

        if not mask_path.exists():

            raise FileNotFoundError(
                f"Masque introuvable : "
                f"{mask_path}"
            )

        try:

            with rasterio.open(mask_path) as src:

                mask = src.read(1)

        except Exception as e:

            raise RuntimeError(
                f"Impossible de lire le masque : "
                f"{mask_path}\n"
                f"Erreur : {e}"
            )

        # ====================================================
        # Nettoyage
        # ====================================================

        mask = np.nan_to_num(
            mask,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # ====================================================
        # Binarisation
        # ====================================================

        mask = (
            mask > 0
        ).astype(
            np.float32
        )

        return mask

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, idx):

        image_path = self.files[idx]

        # ----------------------------------------------------
        # Charger image
        # ----------------------------------------------------

        image = self._load_image(
            image_path
        )

        # ----------------------------------------------------
        # Charger masque
        # ----------------------------------------------------

        mask = self._load_mask(
            image_path
        )

        # ----------------------------------------------------
        # C,H,W -> H,W,C
        #
        # Nécessaire pour Albumentations
        # ----------------------------------------------------

        image_hwc = np.transpose(
            image,
            (1, 2, 0)
        )

        # ----------------------------------------------------
        # Augmentation
        # ----------------------------------------------------

        if self.transform is not None:

            augmented = self.transform(
                image=image_hwc,
                mask=mask,
            )

            image_hwc = augmented[
                "image"
            ]

            mask = augmented[
                "mask"
            ]

        # ----------------------------------------------------
        # H,W,C -> C,H,W
        # ----------------------------------------------------

        image = np.transpose(
            image_hwc,
            (2, 0, 1)
        ).copy()

        # ----------------------------------------------------
        # Torch
        # ----------------------------------------------------

        image = torch.from_numpy(
            image
        ).float()

        mask = torch.from_numpy(
            mask.copy()
        ).float()

        # ----------------------------------------------------
        # Ajouter canal au masque
        #
        # H,W
        # ->
        # 1,H,W
        # ----------------------------------------------------

        mask = mask.unsqueeze(0)

        return image, mask


# ============================================================
# DATALOADERS
# ============================================================

def build_dataloaders(cfg):

    data_dir = cfg[
        "data"
    ][
        "data_dir"
    ]

    n_bands = cfg[
        "data"
    ][
        "n_bands"
    ]

    batch_size = cfg[
        "data"
    ][
        "batch_size"
    ]

    num_workers = cfg[
        "data"
    ][
        "num_workers"
    ]

    # ========================================================
    # AUGMENTATION TRAIN
    # ========================================================

    train_transform = (
        get_train_augmentation(cfg)
    )

    # ========================================================
    # TRAIN
    # ========================================================

    train_ds = FieldSegmentationDataset(
        data_dir=data_dir,
        split="train",
        transform=train_transform,
        n_bands=n_bands,
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    val_ds = FieldSegmentationDataset(
        data_dir=data_dir,
        split="val",
        transform=None,
        n_bands=n_bands,
    )

    # ========================================================
    # TEST
    # ========================================================

    test_ds = FieldSegmentationDataset(
        data_dir=data_dir,
        split="test",
        transform=None,
        n_bands=n_bands,
    )

    # ========================================================
    # DATALOADER TRAIN
    # ========================================================

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    # ========================================================
    # DATALOADER VALIDATION
    # ========================================================

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # ========================================================
    # DATALOADER TEST
    # ========================================================

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )