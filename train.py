"""
Script d'entraînement du modèle de segmentation de parcelles agricoles.

Usage:
    python train.py --config config.yaml --data_dir data/AI4Boundaries --epochs 1
"""

import os
import csv
import argparse

import yaml
import torch
import numpy as np

from models.unet import build_model
from data.dataset import build_dataloaders
from utils.losses import get_loss_fn
from utils.metrics import compute_metrics, AverageMeter
from utils.visualize import save_prediction_grid, plot_training_curves


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    loss_meter = AverageMeter()

    total_batches = len(loader)

    for batch_idx, (images, masks) in enumerate(loader, start=1):

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = loss_fn(logits, masks)

        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

        # Affichage toutes les 10 batches
        if batch_idx % 10 == 0 or batch_idx == total_batches:
            print(
                f"  Train batch {batch_idx}/{total_batches} "
                f"- loss={loss_meter.avg:.4f}"
            )

    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, loss_fn, device):

    model.eval()

    loss_meter = AverageMeter()

    metric_meters = {
        k: AverageMeter()
        for k in ["iou", "dice", "precision", "recall"]
    }

    last_batch = None

    total_batches = len(loader)

    for batch_idx, (images, masks) in enumerate(loader, start=1):

        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        loss = loss_fn(logits, masks)

        loss_meter.update(
            loss.item(),
            images.size(0)
        )

        metrics = compute_metrics(
            logits,
            masks
        )

        for k, v in metrics.items():
            metric_meters[k].update(
                v,
                images.size(0)
            )

        last_batch = (
            images,
            masks,
            logits
        )

        if batch_idx % 10 == 0 or batch_idx == total_batches:
            print(
                f"  Val batch {batch_idx}/{total_batches} "
                f"- loss={loss_meter.avg:.4f}"
            )

    avg_metrics = {
        k: m.avg
        for k, m in metric_meters.items()
    }

    return (
        loss_meter.avg,
        avg_metrics,
        last_batch
    )


def main(args):

    # ========================================================
    # Configuration
    # ========================================================

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir

    if args.epochs:
        cfg["training"]["epochs"] = args.epochs

    # ========================================================
    # Seed
    # ========================================================

    set_seed(
        cfg["training"]["seed"]
    )

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device utilisé : {device}"
    )

    # ========================================================
    # Dossiers
    # ========================================================

    os.makedirs(
        cfg["paths"]["checkpoint_dir"],
        exist_ok=True
    )

    os.makedirs(
        cfg["paths"]["output_dir"],
        exist_ok=True
    )

    # ========================================================
    # Dataset
    # ========================================================

    print("\n========== DATASET ==========")

    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    print(
        f"Train : {len(train_loader.dataset)}"
    )

    print(
        f"Val   : {len(val_loader.dataset)}"
    )

    print(
        f"Test  : {len(test_loader.dataset)}"
    )

    # ========================================================
    # Modèle
    # ========================================================

    print("\n========== MODÈLE ==========")

    model = build_model(cfg).to(device)

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"U-Net : {n_params:,} paramètres"
    )

    # ========================================================
    # Optimiseur
    # ========================================================

    lr = cfg["training"]["learning_rate"]

    wd = cfg["training"]["weight_decay"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd
    )

    # ========================================================
    # Scheduler
    # ========================================================

    scheduler_type = cfg["training"]["scheduler"]

    if scheduler_type == "cosine":

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["training"]["epochs"]
        )

    elif scheduler_type == "plateau":

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=5,
            factor=0.5
        )

    else:

        scheduler = None

    # ========================================================
    # Loss
    # ========================================================

    loss_fn = get_loss_fn(
        cfg["training"]["loss"]
    )

    # ========================================================
    # Historique
    # ========================================================

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_dice": []
    }

    best_iou = 0.0

    patience_counter = 0

    # ========================================================
    # Log CSV
    # ========================================================

    log_path = cfg["paths"]["log_file"]

    log_dir = os.path.dirname(log_path)

    if log_dir:
        os.makedirs(
            log_dir,
            exist_ok=True
        )

    with open(
        log_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_iou",
            "val_dice",
            "val_precision",
            "val_recall"
        ])

    # ========================================================
    # ENTRAÎNEMENT
    # ========================================================

    epochs = cfg["training"]["epochs"]

    print("\n========== ENTRAÎNEMENT ==========")

    for epoch in range(
        1,
        epochs + 1
    ):

        print(
            f"\n--- Époque {epoch}/{epochs} ---"
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss, val_metrics, last_batch = validate(
            model,
            val_loader,
            loss_fn,
            device
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        if scheduler is not None:

            if scheduler_type == "plateau":

                scheduler.step(
                    val_metrics["iou"]
                )

            else:

                scheduler.step()

        # ----------------------------------------------------
        # Historique
        # ----------------------------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_iou"].append(
            val_metrics["iou"]
        )

        history["val_dice"].append(
            val_metrics["dice"]
        )

        # ----------------------------------------------------
        # Résultats
        # ----------------------------------------------------

        print(
            "\nRésultats :"
        )

        print(
            f"  Train loss : {train_loss:.4f}"
        )

        print(
            f"  Val loss   : {val_loss:.4f}"
        )

        print(
            f"  Val IoU    : {val_metrics['iou']:.4f}"
        )

        print(
            f"  Val Dice   : {val_metrics['dice']:.4f}"
        )

        print(
            f"  Precision  : {val_metrics['precision']:.4f}"
        )

        print(
            f"  Recall     : {val_metrics['recall']:.4f}"
        )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        with open(
            log_path,
            "a",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch,
                train_loss,
                val_loss,
                val_metrics["iou"],
                val_metrics["dice"],
                val_metrics["precision"],
                val_metrics["recall"]
            ])

        # ----------------------------------------------------
        # Meilleur modèle
        # ----------------------------------------------------

        if val_metrics["iou"] > best_iou:

            best_iou = val_metrics["iou"]

            patience_counter = 0

            ckpt_path = os.path.join(
                cfg["paths"]["checkpoint_dir"],
                "best_model.pt"
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "config":
                        cfg,

                    "epoch":
                        epoch,

                    "val_iou":
                        best_iou
                },
                ckpt_path
            )

            print(
                f"\n  ✅ Nouveau meilleur modèle !"
            )

            print(
                f"  IoU = {best_iou:.4f}"
            )

            print(
                f"  Sauvegardé : {ckpt_path}"
            )

            # ------------------------------------------------
            # Prédictions
            # ------------------------------------------------

            if last_batch is not None:

                images, masks, logits = last_batch

                prediction_dir = os.path.join(
                    cfg["paths"]["output_dir"],
                    "val_predictions"
                )

                os.makedirs(
                    prediction_dir,
                    exist_ok=True
                )

                save_prediction_grid(
                    images,
                    masks,
                    logits,
                    os.path.join(
                        prediction_dir,
                        f"epoch_{epoch:03d}.png"
                    )
                )

        else:

            patience_counter += 1

            print(
                f"  Pas d'amélioration "
                f"({patience_counter}/"
                f"{cfg['training']['early_stopping_patience']})"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if patience_counter >= cfg["training"]["early_stopping_patience"]:

            print(
                f"\n⏹️ Early stopping à l'époque {epoch}"
            )

            break

    # ========================================================
    # Courbes
    # ========================================================

    plot_training_curves(
        history,
        os.path.join(
            cfg["paths"]["output_dir"],
            "training_curves.png"
        )
    )

    # ========================================================
    # FIN
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "🎉 ENTRAÎNEMENT TERMINÉ"
    )

    print(
        f"Meilleur IoU validation : {best_iou:.4f}"
    )

    print(
        "Modèle : "
        f"{cfg['paths']['checkpoint_dir']}/best_model.pt"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=
        "Entraîne un U-Net pour la segmentation "
        "de parcelles agricoles"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Écrase data_dir du config.yaml"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Écrase epochs du config.yaml"
    )

    args = parser.parse_args()

    main(args)