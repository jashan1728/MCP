"""Dataset for training decoder on camouflaged/original image pairs."""

import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class CamouflagePairDataset(Dataset):
    """Dataset of (camouflaged, original) image pairs.

    Expects two directories with matching filenames:
    - orig_dir/xxx.png contains original image
    - cam_dir/xxx.png contains camouflaged version
    """

    def __init__(self, orig_dir, cam_dir, transform=None):
        """Initialize dataset.

        Args:
            orig_dir: Directory containing original images
            cam_dir: Directory containing camouflaged images
            transform: Optional transform to apply to both images
        """
        # Collect all image paths
        self.orig_paths = sorted(
            glob(os.path.join(orig_dir, "*.png")) +
            glob(os.path.join(orig_dir, "*.jpg")) +
            glob(os.path.join(orig_dir, "*.jpeg"))
        )
        self.cam_paths = sorted(
            glob(os.path.join(cam_dir, "*.png")) +
            glob(os.path.join(cam_dir, "*.jpg")) +
            glob(os.path.join(cam_dir, "*.jpeg"))
        )

        assert len(self.orig_paths) == len(self.cam_paths), \
            f"Mismatch: {len(self.orig_paths)} originals vs {len(self.cam_paths)} camouflaged"

        self.transform = transform
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.orig_paths)

    def __getitem__(self, idx):
        """Get a (camouflaged, original) pair.

        Args:
            idx: Index

        Returns:
            cam: Camouflaged image tensor (3, H, W)
            orig: Original image tensor (3, H, W)
        """
        # Load images
        orig = Image.open(self.orig_paths[idx]).convert("RGB")
        cam = Image.open(self.cam_paths[idx]).convert("RGB")

        # Apply transforms if specified
        if self.transform:
            orig = self.transform(orig)
            cam = self.transform(cam)

        # Convert to tensors
        orig = self.to_tensor(orig)  # (3, H, W) in [0, 1]
        cam = self.to_tensor(cam)

        return cam, orig
