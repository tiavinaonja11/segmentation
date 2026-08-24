"""
U-Net pour la segmentation de parcelles agricoles sur images satellite multi-bandes.

Référence : Ronneberger et al., 2015, "U-Net: Convolutional Networks for
Biomedical Image Segmentation".
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) x 2"""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downscaling avec maxpool puis double conv"""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upscaling puis double conv, avec skip connection"""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Ajustement si les dimensions ne correspondent pas exactement
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net configurable en profondeur, adapté aux images multi-bandes
    (ex: Sentinel-2 : R, G, B, NIR, ou davantage de bandes).

    Args:
        n_bands: nombre de canaux d'entrée (bandes spectrales)
        n_classes: nombre de classes de sortie (1 = binaire, >1 = multi-classes)
        base_channels: nombre de filtres du premier niveau
        depth: nombre de niveaux d'encodeur/décodeur
        dropout: taux de dropout dans les blocs convolutifs
    """

    def __init__(self, n_bands=4, n_classes=1, base_channels=32, depth=4, dropout=0.1):
        super().__init__()
        self.n_bands = n_bands
        self.n_classes = n_classes
        self.depth = depth

        self.inc = DoubleConv(n_bands, base_channels, dropout)

        # Encodeur
        self.downs = nn.ModuleList()
        ch = base_channels
        for _ in range(depth):
            self.downs.append(Down(ch, ch * 2, dropout))
            ch *= 2

        # Décodeur
        self.ups = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(Up(ch, ch // 2, dropout))
            ch //= 2

        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        skips = []
        x = self.inc(x)
        skips.append(x)

        for down in self.downs[:-1]:
            x = down(x)
            skips.append(x)
        x = self.downs[-1](x)  # dernier niveau, pas de skip stocké après

        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip)

        return self.outc(x)


def build_model(cfg):
    """Construit le modèle à partir du dictionnaire de config."""
    return UNet(
        n_bands=cfg["data"]["n_bands"],
        n_classes=cfg["data"]["n_classes"],
        base_channels=cfg["model"]["base_channels"],
        depth=cfg["model"]["depth"],
        dropout=cfg["model"]["dropout"],
    )


if __name__ == "__main__":
    # Test rapide de l'architecture
    model = UNet(n_bands=4, n_classes=1, base_channels=32, depth=4)
    x = torch.randn(2, 4, 256, 256)
    y = model(x)
    print(f"Entrée : {x.shape}  ->  Sortie : {y.shape}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre de paramètres : {n_params:,}")
