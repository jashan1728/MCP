"""Main camouflage generation script using Jadena noise burnishing technique."""

import os
import argparse
import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights

# Import from local modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sender.exposure import PolynomialExposure, compute_camouflage_loss
from sender.utils import load_image, save_tensor_as_image


def generate_camouflaged_image(
    model,
    img,
    num_steps=50,
    epsilon=0.2,
    step_size=0.02,
    degree=2,
    save_intermediates=False,
    save_every=10,
    return_intermediates=False,
    intermediate_dir=None,
    base_name="camouflaged"
):
    """Generate camouflaged image using Jadena optimization.

    Args:
        model: Feature extractor (ResNet)
        img: Input image tensor (1, 3, H, W)
        num_steps: Number of optimization steps
        epsilon: L_inf constraint on noise
        step_size: Learning rate for optimization
        degree: Polynomial degree for exposure
        save_intermediates: Whether to save intermediate images
        save_every: Save intermediate every N steps
        return_intermediates: Whether to return list of intermediates
        intermediate_dir: Directory to save intermediate images
        base_name: Base name for intermediate files

    Returns:
        final_img: Final camouflaged image (1, 3, H, W)
        intermediates: List of intermediate images (if return_intermediates=True)
    """
    device = img.device
    B, C, H, W = img.shape

    # Initialize exposure and noise parameters
    exposure = PolynomialExposure(degree=degree, H=H, W=W).to(device)
    noise = torch.zeros_like(img, requires_grad=True)

    params = list(exposure.parameters()) + [noise]
    optimizer = torch.optim.Adam(params, lr=step_size)

    intermediates = []

    print(f"Starting camouflage optimization for {num_steps} steps...")

    for t in range(num_steps):
        optimizer.zero_grad()

        # Apply exposure and add noise
        I_cam = exposure(img) + noise
        I_cam = torch.clamp(I_cam, 0.0, 1.0)

        # Compute loss
        loss = compute_camouflage_loss(model, img, I_cam, exposure, noise, epsilon)
        loss.backward()
        optimizer.step()

        # Project noise to L_inf ball
        with torch.no_grad():
            noise.clamp_(-epsilon, epsilon)

        if (t + 1) % 10 == 0:
            print(f"  Step {t+1}/{num_steps}, Loss: {loss.item():.4f}")

        # Save or collect intermediate snapshots
        if save_intermediates and ((t + 1) % save_every == 0 or t == num_steps - 1):
            snap = I_cam.detach().cpu().clone()

            if return_intermediates:
                intermediates.append(snap)

            if intermediate_dir is not None:
                os.makedirs(intermediate_dir, exist_ok=True)
                step_id = t + 1
                out_path = os.path.join(
                    intermediate_dir,
                    f"{base_name}_step{step_id:03d}.png",
                )
                save_tensor_as_image(snap, out_path)
                print(f"    Saved intermediate: {out_path}")

    final_img = I_cam.detach()

    if return_intermediates:
        return final_img, intermediates
    return final_img


def main():
    """Command-line interface for camouflage generation."""
    parser = argparse.ArgumentParser(description="Generate camouflaged images using Jadena technique")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output camouflaged image path")
    parser.add_argument("--steps", type=int, default=80, help="Number of optimization steps")
    parser.add_argument("--epsilon", type=float, default=0.3, help="L_inf noise constraint")
    parser.add_argument("--save-intermediates", action="store_true", help="Save intermediate images")
    parser.add_argument("--save-every", type=int, default=10, help="Save intermediate every N steps")
    parser.add_argument("--intermediate-dir", default=None, help="Directory for intermediate images")
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load feature extractor
    print("Loading ResNet-50 feature extractor...")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device).eval()

    # Load input image
    print(f"Loading image: {args.input}")
    img = load_image(args.input).to(device)
    print(f"Image shape: {img.shape}")

    # Generate camouflaged image
    base_name = os.path.splitext(os.path.basename(args.output))[0]

    I_cam = generate_camouflaged_image(
        model,
        img,
        num_steps=args.steps,
        epsilon=args.epsilon,
        save_intermediates=args.save_intermediates,
        save_every=args.save_every,
        return_intermediates=False,
        intermediate_dir=args.intermediate_dir,
        base_name=base_name,
    )

    # Save final output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_tensor_as_image(I_cam.cpu(), args.output)
    print(f"\n✓ Saved camouflaged image: {args.output}")


if __name__ == "__main__":
    main()
