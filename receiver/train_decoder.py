"""Training script for the decoder network."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from receiver.decoder_model import CamouflageDecoder
from receiver.decoder_data import CamouflagePairDataset


def train_decoder(
    orig_dir="data/original",
    cam_dir="data/camouflaged",
    save_path="decoder.pth",
    batch_size=8,
    num_epochs=50,
    lr=1e-4,
    device=None,
):
    """Train the decoder network.

    Args:
        orig_dir: Directory containing original images
        cam_dir: Directory containing camouflaged images
        save_path: Path to save trained model
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on (auto-detect if None)

    Returns:
        Trained model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training decoder on device: {device}")
    print(f"Loading data from:")
    print(f"  Original: {orig_dir}")
    print(f"  Camouflaged: {cam_dir}")

    # Create dataset and dataloader
    dataset = CamouflagePairDataset(orig_dir, cam_dir)
    print(f"Dataset size: {len(dataset)} image pairs")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )

    # Initialize model
    model = CamouflageDecoder().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.L1Loss()  # Can also use MSELoss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nStarting training for {num_epochs} epochs...")
    print("-" * 60)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (cam, orig) in enumerate(loader):
            cam = cam.to(device)
            orig = orig.to(device)

            # Forward pass
            optimizer.zero_grad()
            recon = model(cam)
            loss = criterion(recon, orig)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * cam.size(0)

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, "
                      f"Batch {batch_idx+1}/{len(loader)}, "
                      f"Loss: {loss.item():.4f}")

        # Epoch statistics
        epoch_loss = running_loss / len(loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {epoch_loss:.4f}")
        print("-" * 60)

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"\n✓ Saved trained decoder to: {save_path}")

    return model


def main():
    """Command-line interface for decoder training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train decoder network")
    parser.add_argument("--orig-dir", default="data/original", help="Original images directory")
    parser.add_argument("--cam-dir", default="data/camouflaged", help="Camouflaged images directory")
    parser.add_argument("--save-path", default="decoder.pth", help="Path to save trained model")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    train_decoder(
        orig_dir=args.orig_dir,
        cam_dir=args.cam_dir,
        save_path=args.save_path,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
