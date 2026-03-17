"""Convolutional autoencoder decoder for reconstructing camouflaged images."""

import torch
import torch.nn as nn


class CamouflageDecoder(nn.Module):
    """Decoder network to reconstruct original images from camouflaged ones.

    Architecture: Convolutional encoder-decoder with 3 downsampling stages.
    Works for any input size where H and W are divisible by 8.
    """

    def __init__(self, in_channels=3, base_channels=64):
        """Initialize decoder.

        Args:
            in_channels: Number of input channels (3 for RGB)
            base_channels: Base number of channels (doubles each stage)
        """
        super().__init__()

        # Encoder - progressively downsample
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )

        # Decoder - progressively upsample
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Output layer
        self.out_conv = nn.Conv2d(base_channels, in_channels, 3, stride=1, padding=1)

    def forward(self, x):
        """Forward pass through decoder.

        Args:
            x: Camouflaged image (B, 3, H, W)

        Returns:
            Reconstructed image (B, 3, H, W) in range [0, 1]
        """
        # Encoder path
        e1 = self.enc1(x)       # (B, 64, H, W)
        e2 = self.enc2(e1)      # (B, 128, H/2, W/2)
        e3 = self.enc3(e2)      # (B, 256, H/4, W/4)
        e4 = self.enc4(e3)      # (B, 512, H/8, W/8)

        # Decoder path
        d1 = self.dec1(e4)      # (B, 256, H/4, W/4)
        d2 = self.dec2(d1)      # (B, 128, H/2, W/2)
        d3 = self.dec3(d2)      # (B, 64, H, W)

        # Output
        out = self.out_conv(d3) # (B, 3, H, W)
        out = torch.sigmoid(out)  # Ensure output in [0, 1]

        return out
