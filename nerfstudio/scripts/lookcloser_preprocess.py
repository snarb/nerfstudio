import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from PIL import Image
from rich.progress import track
from typing_extensions import Literal

try:
    import tinycudann as tcnn
except ImportError:
    print("Error: tinycudann is not installed. Please install it to use LookCloser.")
    sys.exit(1)

from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.utils.rich_utils import CONSOLE


# --- 1. Robust SSIM Implementation ---
def compute_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True):
    """
    Computes SSIM with standard constants for float [0,1] images.
    Matches the "SSIM Loss" metric used in Fig 3(d) of the paper.

    Args:
        img1: (B, C, H, W) tensor
        img2: (B, C, H, W) tensor
    """
    L = 1.0
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    mu1 = F.avg_pool2d(img1, window_size, 1, window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, 1, window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, 1, window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, 1, window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)


# --- 2. 2D NGP Model ---
class InstantNGP2D(nn.Module):
    def __init__(
            self,
            n_levels: int = 16,
            n_features: int = 2,
            min_res: int = 16,
            max_res: int = 2048,
            log2_hashmap_size: int = 19
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_features = n_features
        self.min_res = min_res

        # Geometric growth factor 'b'
        self.b = np.exp((np.log(max_res) - np.log(min_res)) / (n_levels - 1))

        self.encoding = tcnn.Encoding(
            n_input_dims=2,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": min_res,
                "per_level_scale": self.b,
            },
        )

        self.decoder = tcnn.Network(
            n_input_dims=n_levels * n_features,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

    def get_resolution_at_level(self, level_idx: int) -> float:
        return float(self.min_res * (self.b ** level_idx))

    def forward(self, uv):
        # uv should be (N, 2) in [0, 1]
        return self.decoder(self.encoding(uv))

    def render_masked(self, uv_coords, max_active_level: int):
        """
        Renders using only levels 0..max_active_level.
        """
        features = self.encoding(uv_coords)  # (N, L*F)

        # Feature layout: Levels are sequential.
        # Level 0 features: indices 0 to n_features-1
        # Level k features: indices k*n_features to (k+1)*n_features - 1
        cutoff = (max_active_level + 1) * self.n_features

        # Create mask
        mask = torch.zeros_like(features)
        mask[:, :cutoff] = 1.0

        return self.decoder(features * mask)


# --- Helper Functions ---
def generate_uv_grid(y, x, H, W, size, device):
    """Generates normalized UV coordinates for a specific patch (pixel centers)."""
    # Pixel centers convention: +0.5
    ys = torch.linspace(y + 0.5, y + size - 0.5, steps=size, device=device)
    xs = torch.linspace(x + 0.5, x + size - 0.5, steps=size, device=device)

    # meshgrid indexing='ij' means ys varies along dim 0, xs along dim 1
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    # Normalize to [0, 1]
    uv = torch.stack([grid_x / W, grid_y / H], dim=-1)
    return uv.reshape(-1, 2)


def train_2d_ngp(image_tensor: torch.Tensor, steps: int = 3000, batch_size: int = 2 ** 14) -> InstantNGP2D:
    """
    Trains InstantNGP2D on a single image.

    Args:
        image_tensor: (H, W, 3) float tensor in [0, 1]
        steps: Number of training steps
        batch_size: Rays per step
    """
    device = image_tensor.device
    H, W, _ = image_tensor.shape

    model = InstantNGP2D(log2_hashmap_size=19).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, eps=1e-15)

    # Pre-calculate full grid UVs isn't feasible for huge images in one go,
    # but for sampling we can just generate random indices.

    for _ in range(steps):
        # Random pixel coordinates
        inds_y = torch.randint(0, H, (batch_size,), device=device)
        inds_x = torch.randint(0, W, (batch_size,), device=device)

        target_rgb = image_tensor[inds_y, inds_x]  # (B, 3)

        # Convert to UV
        uv = torch.stack([(inds_x + 0.5) / W, (inds_y + 0.5) / H], dim=-1)  # (B, 2)

        optimizer.zero_grad()
        pred_rgb = model(uv)
        loss = F.mse_loss(pred_rgb, target_rgb)
        loss.backward()
        optimizer.step()

    return model


def estimate_frequency_map(
        model: InstantNGP2D,
        image_tensor: torch.Tensor,
        ssim_threshold: float = 0.95,
        patch_size: int = 32
) -> torch.Tensor:
    """
    Computes dense frequency map for a single image.
    Returns: (H // stride, W // stride) tensor of float resolutions.
    """
    stride = patch_size
    H, W, _ = image_tensor.shape

    # Result map size
    h_steps = (H - patch_size) // stride + 1
    w_steps = (W - patch_size) // stride + 1

    freq_map = torch.zeros((h_steps, w_steps), dtype=torch.float32, device=image_tensor.device)

    # Iterate patches
    # Note: To speed this up, one could batch patches, but for clarity/VRAM safety we do strict loops or small batches.
    # We will process row by row.

    y_starts = range(0, H - patch_size + 1, stride)
    x_starts = range(0, W - patch_size + 1, stride)

    for i, y in enumerate(y_starts):
        for j, x in enumerate(x_starts):

            # 1. Get GT Patch
            patch_gt = image_tensor[y: y + patch_size, x: x + patch_size]  # (P, P, 3)

            # Format for SSIM: (1, 3, P, P)
            patch_gt_fmt = patch_gt.permute(2, 0, 1).unsqueeze(0)

            # 2. Get UV Grid for this patch
            uv_grid = generate_uv_grid(y, x, H, W, patch_size, device=image_tensor.device)  # (P*P, 2)

            # 3. Progressive Test
            found_level = False
            for level in range(model.n_levels):
                # Render masked
                patch_pred_flat = model.render_masked(uv_grid, max_active_level=level)  # (P*P, 3)
                patch_pred = patch_pred_flat.view(patch_size, patch_size, 3)

                # Format
                patch_pred_fmt = patch_pred.permute(2, 0, 1).unsqueeze(0)  # (1, 3, P, P)

                # SSIM Check
                score = compute_ssim(patch_gt_fmt, patch_pred_fmt)

                if score > ssim_threshold:
                    freq_map[i, j] = model.get_resolution_at_level(level)
                    found_level = True
                    break

            if not found_level:
                # If even max level doesn't satisfy, assign max resolution
                freq_map[i, j] = model.get_resolution_at_level(model.n_levels - 1)

    return freq_map


@dataclass
class LookCloserPreprocessConfig:
    """Configuration for LookCloser preprocessing script."""

    dataparser: AnnotatedDataParserUnion
    """Data parser config to load the dataset."""

    output_name: str = "lookcloser_frequencies"
    """Name of the output directory (created inside the data directory)."""

    steps_per_image: int = 3000
    """Training steps for the per-image 2D NGP."""

    ssim_threshold: float = 0.95
    """SSIM threshold to determine necessary frequency level."""

    patch_size: int = 32
    """Patch size for frequency analysis."""

    device: Literal["cpu", "cuda"] = "cuda"
    """Compute device."""

    def main(self):
        CONSOLE.print("[bold green]Starting LookCloser Pre-processing...[/bold green]")

        # 1. Setup Data Parser
        dataparser = self.dataparser.setup()
        outputs = dataparser.get_dataparser_outputs(split="train")

        CONSOLE.print(f"Loaded {len(outputs.image_filenames)} training images.")

        # 2. Setup Output Directory
        data_dir = self.dataparser.data
        output_dir = data_dir / self.output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        CONSOLE.print(f"Saving frequency maps to: {output_dir}")

        # 3. Processing Loop
        # We save each map individually as a .pt file with the same stem as the image

        device = torch.device(self.device)

        with torch.no_grad():
            for idx, img_path in enumerate(track(outputs.image_filenames, description="Processing Images")):
                # Check if already exists
                save_path = output_dir / f"{img_path.stem}.pt"
                if save_path.exists():
                    continue

                # Load and preprocess image
                # Dataparser outputs are paths, we need to load them.
                pil_image = Image.open(img_path)
                pil_image = TF.to_tensor(pil_image).permute(1, 2, 0)  # (H, W, 3)

                # Handle alpha if present (RGBA -> RGB white bg or discard alpha)
                if pil_image.shape[-1] == 4:
                    pil_image = pil_image[:, :, :3] * pil_image[:, :, 3:4] + (1 - pil_image[:, :, 3:4])

                pil_image = pil_image.to(device)

                # Train 2D NGP
                with torch.enable_grad():
                    model_2d = train_2d_ngp(pil_image, steps=self.steps_per_image)

                # Estimate Frequencies
                freq_map = estimate_frequency_map(
                    model_2d,
                    pil_image,
                    ssim_threshold=self.ssim_threshold,
                    patch_size=self.patch_size
                )

                # Save
                torch.save(freq_map.cpu(), save_path)

                # Cleanup to prevent VRAM accumulation
                del model_2d
                del pil_image
                del freq_map
                torch.cuda.empty_cache()

        CONSOLE.print("[bold green]Pre-processing complete![/bold green]")
        CONSOLE.print(f"You can now run training with [yellow]ns-train lookcloser --data {data_dir}[/yellow]")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(LookCloserPreprocessConfig).main()


if __name__ == "__main__":
    entrypoint()