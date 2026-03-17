"""Sender module for camouflage generation."""

from .exposure import PolynomialExposure, compute_camouflage_loss
from .utils import load_image, save_tensor_as_image
from .jadena_camouflage import generate_camouflaged_image

__all__ = [
    "PolynomialExposure",
    "compute_camouflage_loss",
    "load_image",
    "save_tensor_as_image",
    "generate_camouflaged_image",
]
