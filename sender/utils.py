"""Utility functions for image loading and saving."""

import torch
import torchvision.transforms as T
from PIL import Image


def load_image(path):
    """Load an image from disk and convert to tensor.

    Args:
        path: Path to image file

    Returns:
        Tensor of shape (1, 3, H, W) in range [0, 1]
    """
    img = Image.open(path).convert("RGB")
    to_tensor = T.ToTensor()
    return to_tensor(img).unsqueeze(0)  # (1, 3, H, W)


def save_tensor_as_image(tensor, path):
    """Save a tensor as an image file.

    Args:
        tensor: Tensor of shape (1, 3, H, W) or (3, H, W) in range [0, 1]
        path: Output file path
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Remove batch dimension
    tensor = torch.clamp(tensor, 0.0, 1.0)
    to_pil = T.ToPILImage()
    img = to_pil(tensor.cpu())
    img.save(path)
