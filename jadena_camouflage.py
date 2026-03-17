import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import math
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Utility: load image / save image
# -----------------------------

# def load_image(path, image_size=256):
#     tfm = transforms.Compose([
#         transforms.Resize((image_size, image_size)),
#         transforms.ToTensor()
#     ])
#     img = Image.open(path).convert("RGB")
#     return tfm(img).unsqueeze(0).to(device)  # (1,3,H,W)

def load_image(path):
    img = Image.open(path).convert("RGB")
    tfm = transforms.ToTensor()
    return tfm(img).unsqueeze(0).to(device)  # (1,3,H,W) with original H,W



def save_image(tensor, path):
    tensor = tensor.detach().clamp(0, 1).cpu()
    tfm = transforms.ToPILImage()
    img = tfm(tensor.squeeze(0))
    img.save(path)


# -----------------------------
# Feature extractor φ(·)
# -----------------------------

class ResNetFeatureExtractor(nn.Module):
    """
    Use intermediate ResNet-50 layers as feature maps φ_j(·),
    similar to paper's high-level feature extraction. [file:37]
    """
    def __init__(self, layers=("layer2", "layer3", "layer4")):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.layers = layers
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        feats = {}
        x = self.stem(x)
        x = self.layer1(x)
        if "layer1" in self.layers:
            feats["layer1"] = x
        x = self.layer2(x)
        if "layer2" in self.layers:
            feats["layer2"] = x
        x = self.layer3(x)
        if "layer3" in self.layers:
            feats["layer3"] = x
        x = self.layer4(x)
        if "layer4" in self.layers:
            feats["layer4"] = x
        return feats


# -----------------------------
# Exposure model θ_e(a, U) (Eq. (5)-style) [file:37]
# -----------------------------

class PolynomialExposure(nn.Module):
    """
    Locally-variant multivariate polynomial exposure in log-domain:
    log(θ_e,p) = Σ_{d=0..D} Σ_{l=0..D-d} a_{d,l} (x_p+u_p)^d (y_p+v_p)^l
    as described in Eq. (5). [file:37]
    """
    def __init__(self, height, width, degree=2):
        super().__init__()
        self.H = height
        self.W = width
        self.D = degree

        # Polynomial coefficients a_{d,l}
        # We flatten (d,l) into one dimension for simplicity
        num_terms = sum((self.D - d + 1) for d in range(self.D + 1))
        self.a = nn.Parameter(torch.zeros(num_terms, device=device))

        # Offset map U = (u_p, v_p), initialized to 0
        self.U = nn.Parameter(torch.zeros(1, 2, height, width, device=device))

        # Precompute coordinate grid (x_p, y_p)
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, steps=height, device=device),
            torch.linspace(-1.0, 1.0, steps=width, device=device),
            indexing="ij"
        )
        self.register_buffer("xs", xs)  # (H,W)
        self.register_buffer("ys", ys)  # (H,W)

        # Precompute powers for efficiency
        self._precompute_powers()

    def _precompute_powers(self):
        # Powers of x, y up to degree D: x^0..x^D etc.
        x_pows = [torch.ones_like(self.xs)]
        y_pows = [torch.ones_like(self.ys)]
        for i in range(1, self.D + 1):
            x_pows.append(x_pows[-1] * self.xs)
            y_pows.append(y_pows[-1] * self.ys)
        self.x_pows = nn.Parameter(torch.stack(x_pows, dim=0), requires_grad=False)
        self.y_pows = nn.Parameter(torch.stack(y_pows, dim=0), requires_grad=False)

    def forward(self, batch_size):
        """
        Return θ_e with shape (B,1,H,W), broadcastable over channels.
        """
        # Compute (x_p + u_p), (y_p + v_p)
        u = self.U[:, 0]  # (1,H,W)
        v = self.U[:, 1]  # (1,H,W)
        x_eff = self.xs + u  # (H,W)
        y_eff = self.ys + v  # (H,W)

        # Compute powers of x_eff, y_eff up to D
        x_pows = [torch.ones_like(x_eff)]
        y_pows = [torch.ones_like(y_eff)]
        for i in range(1, self.D + 1):
            x_pows.append(x_pows[-1] * x_eff)
            y_pows.append(y_pows[-1] * y_eff)

        # Compute polynomial sum over (d,l)
        log_theta = torch.zeros_like(x_eff)
        idx = 0
        for d in range(self.D + 1):
            for l in range(self.D - d + 1):
                coeff = self.a[idx]
                term = coeff * (x_pows[d] * y_pows[l])
                log_theta = log_theta + term
                idx += 1

        # Expand to (B,1,H,W) and exponentiate
        # log_theta is currently (H,W), (1,H,W), or possibly already (1,1,H,W)
        if log_theta.dim() == 2:          # (H,W)
            log_theta = log_theta.unsqueeze(0).unsqueeze(0)   # -> (1,1,H,W)
        elif log_theta.dim() == 3:        # (1,H,W)
            log_theta = log_theta.unsqueeze(1)                # -> (1,1,H,W)
        # If it's already (1,1,H,W), leave it as is

        # Now expand to (B,1,H,W) for the batch
        log_theta = log_theta.expand(batch_size, -1, -1, -1)  # (B,1,H,W)
        theta_e = torch.exp(log_theta)
        return theta_e



# -----------------------------
# Losses: J_smooth and J_co-cons [file:37]
# -----------------------------

def smoothness_loss(theta_e_log, U, lambda_b=1e-3, lambda_s=1e-3):
    """
    Jsmooth(a, U) = -λ_b ||log(θ_e)||_2^2 - λ_s ||∇U||_2^2  (Eq. (6)). [file:37]
    We will *maximize* J_smooth, but in PyTorch we usually minimize,
    so we return -Jsmooth here (i.e., a penalty).
    """
    # theta_e_log: (B,1,H,W)
    # U: (1,2,H,W)
    l2_theta = torch.mean(theta_e_log ** 2)

    # Total variation on U
    def tv(x):
        dx = x[:, :, 1:, :] - x[:, :, :-1, :]
        dy = x[:, :, :, 1:] - x[:, :, :, :-1]
        return (dx ** 2).mean() + (dy ** 2).mean()

    tv_U = tv(U)
    Jsmooth = -lambda_b * l2_theta - lambda_s * tv_U
    # Return negativeJsmooth as a penalty to minimize
    return -Jsmooth


def co_consistency_loss(images, feature_extractor, layers=("layer2", "layer3", "layer4")):
    """
    Jco-cons(a,U,θ_n) = -avg( std(Φ_j(R)) ), where Φ_j(R) is channel-wise
    concatenation of feature maps of group R (Eq. (8)). [file:37]

    Here we implement the negative (penalty) so that minimizing it
    corresponds to maximizing Jco-cons.
    """
    # images: list of tensors (B=1,3,H,W), one of which is the adversarial image
    feats_concat = {l: [] for l in layers}

    for img in images:
        feats = feature_extractor(img)
        for l in layers:
            feats_concat[l].append(feats[l])

    # Channel-wise concatenation across group
    std_list = []
    for l in layers:
        # feats_concat[l] : list of (1,C,H,W)
        cat = torch.cat(feats_concat[l], dim=2)  # concat along H (spatial)
        # Compute std over spatial dims, then average over channels
        # shape: (1,C,H',W')
        c = cat
        # Spatial mean per channel
        mean = c.mean(dim=(2, 3), keepdim=True)
        var = ((c - mean) ** 2).mean(dim=(2, 3), keepdim=True)
        std = torch.sqrt(var + 1e-6)  # (1,C,1,1)
        std_list.append(std.mean())   # scalar

    Jco = -torch.mean(torch.stack(std_list))
    # Return penalty to minimize (negative of Jco)
    return -Jco

def save_tensor_as_image(tensor, path):
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = torch.clamp(tensor, 0.0, 1.0)
    to_pil = torchvision.transforms.ToPILImage()
    img = to_pil(tensor.cpu())
    img.save(path)




# -----------------------------
# Main optimization loop (MI-FGSM-style) [file:37]
# -----------------------------

@torch.no_grad()
def generate_camouflaged_image(
    model,
    img,                    # tensor (1, 3, H, W)
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
    device = img.device
    B, C, H, W = img.shape

    exposure = PolynomialExposure(degree=degree, H=H, W=W).to(device)
    noise = torch.zeros_like(img, requires_grad=True)

    params = list(exposure.parameters()) + [noise]
    optimizer = torch.optim.Adam(params, lr=step_size)

    intermediates = []

    for t in range(num_steps):
        optimizer.zero_grad()

        I_cam = exposure(img) + noise
        I_cam = torch.clamp(I_cam, 0.0, 1.0)

        loss = compute_camouflage_loss(model, img, I_cam, exposure, noise, epsilon)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            noise.clamp_(-epsilon, epsilon)

        # save or collect intermediate frames
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

    final_img = I_cam.detach()
    if return_intermediates:
        return final_img, intermediates
    return final_img


# -----------------------------
# Example usage
# -----------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="camouflaged.png", help="Path to save camouflaged image")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--step_size", type=float, default=0.02)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Load image
    img = load_image(args.input)

    # Feature extractor φ(·)
    feat_extractor = ResNetFeatureExtractor().to(device)
    feat_extractor.eval()

    # Generate camouflaged image
    with torch.no_grad():
        pass  # just to be explicit: we will enable grads inside function

    I_cam = generate_camouflaged_image(
        img,
        feat_extractor,
        num_steps=args.steps,
        epsilon=args.epsilon,
        step_size=args.step_size,
        degree=2,
        lambda_b=1e-3,
        lambda_s=1e-3,
        lambda_co=1.0,
        use_group_aug=True
    )

    # Save
    save_image(I_cam, args.output)
    print(f"Saved camouflaged image to {args.output}")
