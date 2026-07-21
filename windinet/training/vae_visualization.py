"""PNG reconstruction panels and epoch-level loss curves for VAE training."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/windinet-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def denormalize_fields(
    tensor: torch.Tensor,
    channel_mean: list[float],
    channel_std: list[float],
    normalization_clip: float,
) -> torch.Tensor:
    """Convert normalized [B,C,F,H,W] fields back to physical values."""
    mean = tensor.new_tensor(channel_mean).view(1, 4, 1, 1, 1)
    scale = tensor.new_tensor(channel_std).view(1, 4, 1, 1, 1) * normalization_clip
    return tensor * scale + mean


def save_reconstruction_panels(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_id: str,
    epoch: int,
    frame_numbers: list[int],
    channel_names: list[str],
    output_dir: str | Path,
    dpi: int,
) -> list[Path]:
    """Save one four-channel GT/prediction/residual panel per requested frame."""
    prediction = prediction.detach().float().cpu()
    target = target.detach().float().cpu()
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
    save_dir = Path(output_dir) / "visualizations" / f"epoch_{epoch:04d}" / safe_id
    save_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    num_frames = target.shape[1]
    for frame_number in frame_numbers:
        frame_index = frame_number - 1
        if frame_index >= num_frames:
            continue

        gt = target[:, frame_index].numpy()
        pred = prediction[:, frame_index].numpy()
        residual = pred - gt
        frame_rmse = float(np.sqrt(np.mean(residual**2)))
        channel_rmse = np.sqrt(np.mean(residual**2, axis=(1, 2)))

        fig, axes = plt.subplots(4, 3, figsize=(12, 13), constrained_layout=True)
        for channel, name in enumerate(channel_names):
            value_min = float(min(gt[channel].min(), pred[channel].min()))
            value_max = float(max(gt[channel].max(), pred[channel].max()))
            residual_limit = max(float(np.abs(residual[channel]).max()), 1e-12)

            images = (
                axes[channel, 0].imshow(gt[channel], cmap="viridis", vmin=value_min, vmax=value_max),
                axes[channel, 1].imshow(pred[channel], cmap="viridis", vmin=value_min, vmax=value_max),
                axes[channel, 2].imshow(
                    residual[channel], cmap="coolwarm", vmin=-residual_limit, vmax=residual_limit
                ),
            )
            axes[channel, 0].set_ylabel(f"{name}\nRMSE={channel_rmse[channel]:.4e}")
            for column, title in enumerate(("GT", "Prediction", "Residual (Pred-GT)")):
                axes[channel, column].set_title(title)
                axes[channel, column].set_xticks([])
                axes[channel, column].set_yticks([])
                fig.colorbar(images[column], ax=axes[channel, column], fraction=0.046, pad=0.04)

        fig.suptitle(
            f"epoch={epoch}  sample={sample_id}  frame={frame_number}  RMSE={frame_rmse:.4e}",
            fontsize=13,
        )
        path = save_dir / f"frame_{frame_number:04d}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        saved.append(path)

    return saved


def save_metrics_history(rows: list[dict[str, float]], output_dir: str | Path) -> tuple[Path, Path]:
    """Write epoch metrics to CSV and update the train/validation loss curve."""
    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "metrics.csv"
    fieldnames = list(rows[0])
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    flat_axes = axes.ravel()
    for axis, metric, title in zip(
        flat_axes[:5],
        ("total_loss", "rmse", "h1", "ssim", "mlw"),
        ("Total reconstruction loss", "RMSE", "H1 semi-norm", "SSIM loss", "Wavelet loss"),
    ):
        axis.plot(epochs, [row[f"train_{metric}"] for row in rows], marker="o", label="train")
        axis.plot(epochs, [row[f"val_{metric}"] for row in rows], marker="o", label="validation")
        axis.set(title=title, xlabel="Epoch", ylabel="Loss")
        axis.grid(alpha=0.3)
        axis.legend()

    flat_axes[5].plot(epochs, [row["val_vrmse"] for row in rows], marker="o", color="tab:red")
    flat_axes[5].set(title="Validation VRMSE", xlabel="Epoch", ylabel="VRMSE")
    flat_axes[5].grid(alpha=0.3)

    curve_path = metrics_dir / "loss_curves.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    return csv_path, curve_path
