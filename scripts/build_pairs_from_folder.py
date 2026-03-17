"""Script to generate camouflaged versions of all images in a folder.

This creates training pairs for the decoder:
- Reads all images from data/original/
- Generates camouflaged versions in data/camouflaged/
"""

import os
import glob
import torch
from torchvision.models import resnet50, ResNet50_Weights

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sender.jadena_camouflage import generate_camouflaged_image
from sender.utils import load_image, save_tensor_as_image


def build_pairs(orig_dir, cam_dir, steps=80, epsilon=0.3, skip_existing=True):
    """Generate camouflaged versions of all images in orig_dir.

    Args:
        orig_dir: Directory containing original images
        cam_dir: Directory to save camouflaged images
        steps: Number of optimization steps per image
        epsilon: L_inf noise constraint
        skip_existing: Skip images that already have camouflaged versions
    """
    os.makedirs(cam_dir, exist_ok=True)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load feature extractor once
    print("Loading ResNet-50 feature extractor...")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device).eval()

    # Get all image paths
    image_paths = sorted(
        glob.glob(os.path.join(orig_dir, "*.png")) +
        glob.glob(os.path.join(orig_dir, "*.jpg")) +
        glob.glob(os.path.join(orig_dir, "*.jpeg"))
    )

    print(f"\nFound {len(image_paths)} images in {orig_dir}")
    print(f"Will save camouflaged versions to {cam_dir}")
    print(f"Settings: steps={steps}, epsilon={epsilon}")
    print("=" * 60)

    # Process each image
    for idx, path in enumerate(image_paths):
        fname = os.path.basename(path)
        fname_png = os.path.splitext(fname)[0] + ".png"
        out_path = os.path.join(cam_dir, fname_png)

        # Skip if already exists
        if skip_existing and os.path.exists(out_path):
            print(f"[{idx+1}/{len(image_paths)}] Skipping {fname} (already exists)")
            continue

        print(f"\n[{idx+1}/{len(image_paths)}] Processing: {fname}")

        # Load and camouflage
        img = load_image(path).to(device)
        print(f"  Image shape: {img.shape}")

        I_cam = generate_camouflaged_image(
            model,
            img,
            num_steps=steps,
            epsilon=epsilon,
            save_intermediates=False,
            return_intermediates=False,
        )

        # Save
        save_tensor_as_image(I_cam.cpu(), out_path)
        print(f"  ✓ Saved: {out_path}")

    print("\n" + "=" * 60)
    print(f"✓ Completed! Processed {len(image_paths)} images.")


def main():
    """Command-line interface for batch camouflage generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate camouflaged training pairs")
    parser.add_argument("--orig-dir", default="data/original", help="Original images directory")
    parser.add_argument("--cam-dir", default="data/camouflaged", help="Output directory for camouflaged images")
    parser.add_argument("--steps", type=int, default=80, help="Optimization steps per image")
    parser.add_argument("--epsilon", type=float, default=0.3, help="L_inf noise constraint")
    parser.add_argument("--no-skip", action="store_true", help="Re-process existing images")
    args = parser.parse_args()

    build_pairs(
        args.orig_dir,
        args.cam_dir,
        steps=args.steps,
        epsilon=args.epsilon,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()
