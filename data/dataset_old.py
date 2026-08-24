"""
Dataset PyTorch pour la segmentation de parcelles agricoles.

Structure attendue du dossier de données :
    data_dir/
        images/   *.tif   (multi-bandes, ex: R,G,B,NIR)
        masks/    *.tif   (masque binaire, même nom de fichier que l'image)
"""

import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

import albumentations as A


def get_train_augmentation(cfg):
    aug_cfg = cfg["augmentation"]
    return A.Compose(
        [
            A.HorizontalFlip(p=aug_cfg["horizontal_flip"]),
            A.VerticalFlip(p=aug_cfg["vertical_flip"]),
            A.RandomRotate90(p=aug_cfg["rotate90"]),
            A.RandomBrightnessContrast(p=aug_cfg["brightness_contrast"]),
            A.GaussNoise(p=aug_cfg["gaussian_noise"]),
        ]
    )


class FieldSegmentationDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None, n_bands=4):
        self.image_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.transform = transform
        self.n_bands = n_bands

        all_files = sorted(glob.glob(os.path.join(self.image_dir, "*.tif")))
        if not all_files:
            all_files = sorted(glob.glob(os.path.join(self.image_dir, "*.npy")))

        # split déterministe 70/15/15
        n = len(all_files)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)

        if split == "train":
            self.files = all_files[:n_train]
        elif split == "val":
            self.files = all_files[n_train:n_train + n_val]
        else:
            self.files = all_files[n_train + n_val:]

        if len(self.files) == 0:
            raise RuntimeError(
                f"Aucun fichier trouvé dans {self.image_dir} pour le split '{split}'. "
                "Avez-vous généré/téléchargé les données ?"
            )

    def __len__(self):
        return len(self.files)

    def _load_array(self, path):
        if path.endswith(".npy"):
            return np.load(path)
        if HAS_RASTERIO:
            with rasterio.open(path) as src:
                arr = src.read()  # (bands, H, W)
            return arr
        raise RuntimeError(
            "rasterio n'est pas installé et le fichier n'est pas un .npy. "
            "Installez rasterio : pip install rasterio"
        )

    def __getitem__(self, idx):
        img_path = self.files[idx]
        fname = os.path.basename(img_path)
        mask_path = os.path.join(self.mask_dir, fname)

        image = self._load_array(img_path).astype(np.float32)  # (C, H, W)
        mask = self._load_array(mask_path).astype(np.float32)  # (1, H, W) ou (H, W)

        if mask.ndim == 3:
            mask = mask[0]

        # normalisation simple par bande (0-1) — adaptez selon vos données réelles
        image = image / max(image.max(), 1e-6)

        # (C, H, W) -> (H, W, C) pour albumentations
        image_hwc = np.transpose(image, (1, 2, 0))

        if self.transform:
            augmented = self.transform(image=image_hwc, mask=mask)
            image_hwc = augmented["image"]
            mask = augmented["mask"]

        image_chw = np.transpose(image_hwc, (2, 0, 1)).copy()

        image_tensor = torch.from_numpy(image_chw).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)

        return image_tensor, mask_tensor


def build_dataloaders(cfg):
    from torch.utils.data import DataLoader

    data_dir = cfg["data"]["data_dir"]
    n_bands = cfg["data"]["n_bands"]
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]

    train_transform = get_train_augmentation(cfg)

    train_ds = FieldSegmentationDataset(data_dir, "train", train_transform, n_bands)
    val_ds = FieldSegmentationDataset(data_dir, "val", None, n_bands)
    test_ds = FieldSegmentationDataset(data_dir, "test", None, n_bands)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
