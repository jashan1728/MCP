"""Inference script for decoder - reconstructs original from camouflaged image."""

import os
import torch
import torchvision.transforms as T
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from receiver.decoder_model import CamouflageDecoder


def load_decoder(checkpoint_path, device=None):
    """Load trained decoder from checkpoint.

    Args:
        checkpoint_path: Path to saved model (.pth file)
        device: Device to load model on (auto-detect if None)

    Returns:
        model: Loaded decoder model in eval mode
        device: Device the model is on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CamouflageDecoder().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"✓ Loaded decoder from: {checkpoint_path}")
    print(f"  Using device: {device}")

    return model, device


def decode_image(decoder, device, cam_path, out_path=None):
    """Decode a camouflaged image back to original.

    Args:
        decoder: Trained decoder model
        device: Device model is on
        cam_path: Path to camouflaged image
        out_path: Optional path to save reconstructed image

    Returns:
        Reconstructed image as PIL Image
    """
    # Load camouflaged image
    img = Image.open(cam_path).convert("RGB")
    to_tensor = T.ToTensor()
    cam = to_tensor(img).unsqueeze(0).to(device)  # (1, 3, H, W)

    # Decode
    with torch.no_grad():
        recon = decoder(cam)

    # Convert back to PIL
    recon = recon.squeeze(0).cpu()  # (3, H, W)
    to_pil = T.ToPILImage()
    out_img = to_pil(recon)

    # Save if output path specified
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out_img.save(out_path)
        print(f"✓ Saved decoded image: {out_path}")

    return out_img


def main():
    """Command-line interface for decoder inference."""
    import argparse

    parser = argparse.ArgumentParser(description="Decode camouflaged image")
    parser.add_argument("--cam", required=True, help="Path to camouflaged image")
    parser.add_argument("--checkpoint", default="decoder.pth", help="Path to decoder checkpoint")
    parser.add_argument("--out", default=None, help="Output path (default: auto-generate)")
    args = parser.parse_args()

    # Load decoder
    decoder, device = load_decoder(args.checkpoint)

    # Auto-generate output path if not specified
    if args.out is None:
        base = os.path.splitext(os.path.basename(args.cam))[0]
        args.out = f"{base}_decoded.png"

    # Decode image
    print(f"Decoding: {args.cam}")
    decode_image(decoder, device, args.cam, args.out)


if __name__ == "__main__":
    main()
