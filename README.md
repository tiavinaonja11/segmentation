# Segmentation des parcelles agricoles (Sentinel-2)

Pipeline complet d'apprentissage profond pour détecter les **limites de parcelles agricoles**
à partir d'images satellite Sentinel-2, basé sur le dataset public **AI4Boundaries** (JRC,
Commission Européenne).

## 📁 Structure du projet

```
agri_field_segmentation/
├── README.md
├── requirements.txt
├── config.yaml                  # tous les hyperparamètres
├── data/
│   ├── download_data.py         # télécharge le dataset AI4Boundaries (Sentinel-2)
│   ├── generate_synthetic_data.py  # jeu de données factice pour tester le pipeline immédiatement
│   └── dataset.py               # classe PyTorch Dataset (lecture GeoTIFF + augmentation)
├── models/
│   └── unet.py                  # architecture U-Net (multi-bandes)
├── utils/
│   ├── losses.py                # Dice Loss / BCE+Dice / Focal
│   ├── metrics.py                # IoU, Dice, précision, rappel
│   └── visualize.py             # visualisation image / masque / prédiction
├── train.py                     # boucle d'entraînement complète
├── inference.py                 # prédiction sur nouvelles images
├── checkpoints/                 # poids du modèle sauvegardés
└── outputs/                     # prédictions générées
```

## 🛰️ Le dataset : AI4Boundaries

- Publié par le **Joint Research Centre (JRC)** de la Commission Européenne (2022).
- 7 831 échantillons de 256×256 pixels en Sentinel-2 (résolution 10 m), avec masques
  binaires de parcelles + masques de bordures, couvrant 7 pays européens.
- Licence ouverte, réutilisation libre.
- Article scientifique : d'Andrimont et al., 2023, *Earth System Science Data*.
- Téléchargement : `https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES/`

➡️ Le script `data/download_data.py` automatise le téléchargement et l'organisation
des fichiers. **Ce sandbox n'a pas accès à ce serveur FTP** (réseau restreint), donc
vous devez exécuter ce script sur votre propre machine.

En attendant, `data/generate_synthetic_data.py` crée un petit jeu de données factice
(images + masques de parcelles géométriques) pour que vous puissiez **tester tout le
pipeline immédiatement**, sans attendre le téléchargement.

## 🚀 Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## 🧪 Étape 1 — Tester le pipeline avec des données factices

```bash
python data/generate_synthetic_data.py --out data/synthetic --n_samples 200
python train.py --config config.yaml --data_dir data/synthetic --epochs 5
```

## 📥 Étape 2 — Télécharger les vraies données AI4Boundaries

```bash
python data/download_data.py --country FR --out data/ai4boundaries
```

Pays disponibles : `AT` (Autriche), `ES` (Catalogne), `FR` (France), `LU` (Luxembourg),
`NL` (Pays-Bas), `SI` (Slovénie), `SE` (Suède).

## 🏋️ Étape 3 — Entraîner sur les vraies données

```bash
python train.py --config config.yaml --data_dir data/ai4boundaries --epochs 100
```

Le script sauvegarde automatiquement :
- le meilleur modèle (`checkpoints/best_model.pt`) selon l'IoU de validation
- les courbes de perte/IoU (`outputs/training_curves.png`)
- des exemples de prédictions (`outputs/val_predictions/`)

## 🔍 Étape 4 — Inférence sur de nouvelles images

```bash
python inference.py --checkpoint checkpoints/best_model.pt --image path/to/image.tif --out outputs/prediction.png
```

## 🧠 Modèle

U-Net "from scratch" en PyTorch, adapté aux images multi-bandes (4 bandes Sentinel-2 :
Rouge, Vert, Bleu, Proche-infrarouge par défaut — configurable), avec :
- Batch normalization + Dropout
- Skip connections classiques
- Sortie sigmoïde (segmentation binaire : parcelle / non-parcelle)
- Facilement extensible en multi-classes (cultures, sol nu, eau...)

## 📊 Métriques

- **IoU (Intersection over Union)**
- **Dice / F1-score**
- **Précision / Rappel** au niveau pixel

## 🔧 Personnalisation

Tout se configure dans `config.yaml` : nombre de bandes, taille de patch, taille de
batch, taux d'apprentissage, nombre d'époques, augmentation de données, etc.

## ⚠️ Notes importantes

- Le code utilise `rasterio` pour lire les GeoTIFF (nécessite GDAL — voir `requirements.txt`).
- Adaptez `N_BANDS` dans `config.yaml` selon vos images (4 bandes RGB+NIR par défaut,
  mais vous pouvez utiliser les 10-13 bandes Sentinel-2 complètes).
- Pour une segmentation multi-classes (types de cultures), changez `N_CLASSES` dans
  `config.yaml` et utilisez `CrossEntropyLoss` au lieu de la Dice/BCE binaire
  (voir commentaires dans `utils/losses.py`).
