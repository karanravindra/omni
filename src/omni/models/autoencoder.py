import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int):
        super().__init__()

        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Conv2d(in_ch, out_ch, kernel_size=1)
        )

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        latent_channels: int,
        groups: int,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down1 = nn.Sequential(
            ResBlock(base_channels, base_channels, groups),
            nn.Conv2d(base_channels, base_channels, 4, stride=2, padding=1),
        )

        self.down2 = nn.Sequential(
            ResBlock(base_channels, base_channels * 2, groups),
            nn.Conv2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1),
        )

        self.down3 = nn.Sequential(
            ResBlock(base_channels * 2, base_channels * 4, groups),
            nn.Conv2d(base_channels * 4, base_channels * 4, 4, stride=2, padding=1),
        )

        self.mid = ResBlock(base_channels * 4, base_channels * 4, groups)

        self.norm_out = nn.GroupNorm(groups, base_channels * 4)
        self.conv_out = nn.Conv2d(base_channels * 4, latent_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.mid(x)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        base_channels: int,
        latent_channels: int,
        groups: int,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(latent_channels, base_channels * 4, 3, padding=1)

        self.mid = ResBlock(base_channels * 4, base_channels * 4, groups)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ResBlock(base_channels * 4, base_channels * 2, groups),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ResBlock(base_channels * 2, base_channels, groups),
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ResBlock(base_channels, base_channels, groups),
        )

        self.norm_out = nn.GroupNorm(groups, base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, z):
        x = self.conv_in(z)
        x = self.mid(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return torch.sigmoid(x)


class Autoencoder(nn.Module):
    """
    SDXL-style spatial autoencoder.

    Defaults:
      - 512x512 images
      - Latents: 4 x 64 x 64
      - GroupNorm + residual blocks
      - SD-compatible latent scaling
    """

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 32,
        latent_channels: int = 4,
        groupnorm_groups: int = 32,
        latent_scale: float = 0.18215,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.latent_scale = latent_scale

        self.encoder = Encoder(
            in_channels=image_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            groups=groupnorm_groups,
        )

        self.decoder = Decoder(
            out_channels=image_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            groups=groupnorm_groups,
        )

    def encode(self, x):
        return self.encoder(x) * self.latent_scale

    def decode(self, z):
        return self.decoder(z / self.latent_scale)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
