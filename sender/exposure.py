"""Polynomial exposure model and camouflage loss computation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolynomialExposure(nn.Module):
    """Polynomial exposure function in log domain.

    Models exposure as: theta_e = exp(sum_i a_i * U^i)
    where U is a learned offset map and a_i are polynomial coefficients.
    """

    def __init__(self, degree=2, H=256, W=256):
        super().__init__()
        self.degree = degree
        self.H = H
        self.W = W

        # Polynomial coefficients
        self.coeffs = nn.Parameter(torch.zeros(degree + 1))
        # Initialize: a_0 = 0 (identity exposure), others near zero
        nn.init.normal_(self.coeffs, mean=0.0, std=0.01)

        # Offset map U (per-pixel)
        self.offset_map = nn.Parameter(torch.zeros(1, 1, H, W))

    def forward(self, img):
        """Apply polynomial exposure to image.

        Args:
            img: Input tensor (B, 3, H, W)

        Returns:
            Exposed image (B, 3, H, W)
        """
        batch_size = img.shape[0]

        # Compute polynomial in log domain
        U = self.offset_map  # (1, 1, H, W)

        log_theta = self.coeffs[0]  # scalar

        for i in range(1, self.degree + 1):
            log_theta = log_theta + self.coeffs[i] * (U ** i)

        # log_theta is (1, 1, H, W), expand to batch
        # Check current dimensionality and add unsqueeze if needed
        while log_theta.dim() < 4:
            log_theta = log_theta.unsqueeze(0)

        log_theta = log_theta.expand(batch_size, 1, self.H, self.W)

        # Convert to linear domain
        theta_e = torch.exp(log_theta)  # (B, 1, H, W)

        # Apply to all channels
        return theta_e * img


def compute_camouflage_loss(model, img_orig, img_cam, exposure, noise, epsilon, lambda_co=1.0, lambda_smooth=0.1):
    """Compute Jadena camouflage loss.

    Args:
        model: Feature extractor (e.g., ResNet)
        img_orig: Original image (B, 3, H, W)
        img_cam: Camouflaged image (B, 3, H, W)
        exposure: PolynomialExposure module
        noise: Additive noise tensor
        epsilon: L_inf constraint on noise
        lambda_co: Weight for co-consistency loss
        lambda_smooth: Weight for smoothness regularization

    Returns:
        Total loss scalar
    """
    # Feature extraction
    with torch.no_grad():
        feat_orig = extract_features(model, img_orig)
    feat_cam = extract_features(model, img_cam)

    # Co-consistency loss: minimize feature similarity
    loss_co = -torch.mean((feat_orig - feat_cam) ** 2)

    # Smoothness regularization
    # 1. Log-domain penalty (Eq. 6 from paper)
    U = exposure.offset_map
    log_penalty = torch.mean(torch.log(1 + U ** 2))

    # 2. Total variation on offset map
    tv_loss = total_variation(U)

    loss_smooth = log_penalty + 0.1 * tv_loss

    # Combine losses
    total_loss = lambda_co * loss_co + lambda_smooth * loss_smooth

    return total_loss


def extract_features(model, img):
    """Extract intermediate features from ResNet.

    Args:
        model: ResNet model
        img: Input image (B, 3, H, W)

    Returns:
        Concatenated features from layer2, layer3, layer4
    """
    # Forward through ResNet layers
    x = model.conv1(img)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)

    feat2 = model.layer2(x)
    feat3 = model.layer3(feat2)
    feat4 = model.layer4(feat3)

    # Global average pooling and concatenate
    feat2_pool = F.adaptive_avg_pool2d(feat2, (1, 1)).flatten(1)
    feat3_pool = F.adaptive_avg_pool2d(feat3, (1, 1)).flatten(1)
    feat4_pool = F.adaptive_avg_pool2d(feat4, (1, 1)).flatten(1)

    features = torch.cat([feat2_pool, feat3_pool, feat4_pool], dim=1)
    return features


def total_variation(img):
    """Compute total variation loss for smoothness.

    Args:
        img: Tensor (B, C, H, W)

    Returns:
        TV loss scalar
    """
    tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w
