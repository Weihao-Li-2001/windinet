#!/usr/bin/env python3
"""
Render ShockWaveNet DiT predictions (scripts/inference_shockwave.py's .npz
output) as GT / Prediction / Residual videos, one per sample.

The inference script only saves the predicted fields -- no ground truth, no
picture. This pulls the matching ground-truth simulation back out of the
source .h5 (same sample id) and animates all four channels frame-by-frame,
same panel layout as vae_visualization.save_reconstruction_panels (GT |
Prediction | Residual columns), so a DiT checkpoint's rollout quality can
actually be looked at instead of just read off a val_loss number.

Usage:
    python scripts/visualize_dit_predictions.py \\
        --pred_dir predictions/ \\
        --h5 euler_mq_dataset/256x256_ds/train.h5 \\
        --out_dir predictions/videos
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/windinet-matplotlib")
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from windinet.training.shockwave_data import CHANNEL_NAMES, ShockWaveDataset


def load_ground_truth(dataset: ShockWaveDataset, sample_id: str) -> dict[str, np.ndarray]:
    idx = dataset.ids.index(sample_id)
    sample = dataset[idx]
    return {name: sample[name].numpy() for name in CHANNEL_NAMES}  # [T,H,W]


def render_video(
    *,
    pred: dict[str, np.ndarray],
    gt: dict[str, np.ndarray],
    sample_id: str,
    gamma: float,
    out_path: Path,
    fps: int,
    dpi: int,
) -> None:
    """One MP4 per sample: 4 rows (channels) x 3 columns (GT/Pred/Residual)."""
    num_frames = min(pred[CHANNEL_NAMES[0]].shape[0], gt[CHANNEL_NAMES[0]].shape[0])

    # Fixed color limits across all frames (per channel) so the colorbar is
    # stable and brightness changes reflect the field, not a rescaled axis.
    limits = {}
    for name in CHANNEL_NAMES:
        g, p = gt[name][:num_frames], pred[name][:num_frames]
        limits[name] = (
            float(min(g.min(), p.min())),
            float(max(g.max(), p.max())),
            float(max(np.abs(p - g).max(), 1e-12)),
        )

    fig, axes = plt.subplots(4, 3, figsize=(12, 13), constrained_layout=True)
    images = [[None, None, None] for _ in CHANNEL_NAMES]
    for row, name in enumerate(CHANNEL_NAMES):
        vmin, vmax, rlim = limits[name]
        for col, title in enumerate(("GT", "Prediction", "Residual (Pred-GT)")):
            cmap = "coolwarm" if col == 2 else "viridis"
            vlo, vhi = (-rlim, rlim) if col == 2 else (vmin, vmax)
            images[row][col] = axes[row, col].imshow(np.zeros_like(gt[name][0]), cmap=cmap, vmin=vlo, vmax=vhi)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(title)
            fig.colorbar(images[row][col], ax=axes[row, col], fraction=0.046, pad=0.04)
        axes[row, 0].set_ylabel(name)

    with imageio.get_writer(out_path, fps=fps) as writer:
        for t in range(num_frames):
            frame_rmse = 0.0
            for row, name in enumerate(CHANNEL_NAMES):
                g, p = gt[name][t], pred[name][t]
                residual = p - g
                frame_rmse += float(np.sqrt(np.mean(residual**2)))
                images[row][0].set_data(g)
                images[row][1].set_data(p)
                images[row][2].set_data(residual)
            fig.suptitle(
                f"{sample_id}  gamma={gamma:.4f}  frame={t + 1}/{num_frames}  "
                f"mean channel RMSE={frame_rmse / len(CHANNEL_NAMES):.4e}",
                fontsize=13,
            )
            fig.canvas.draw()
            frame_rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(frame_rgba[..., :3])

    plt.close(fig)
    print(f"  {sample_id}: {num_frames} frames -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=Path, required=True,
                    help="Directory of .npz files from scripts/inference_shockwave.py")
    ap.add_argument("--h5", type=Path, required=True,
                    help="ShockWave HDF5 file the predictions' initial conditions came from")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Output directory for .mp4 files (default: <pred_dir>/videos)")
    ap.add_argument("--sample_ids", type=str, default=None,
                    help="Comma-separated subset of sample ids (default: every .npz in pred_dir)")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()

    out_dir = args.out_dir or (args.pred_dir / "videos")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_ids:
        npz_paths = [args.pred_dir / f"{sid}.npz" for sid in args.sample_ids.split(",")]
    else:
        npz_paths = sorted(args.pred_dir.glob("*.npz"))

    if not npz_paths:
        raise SystemExit(f"No .npz predictions found in {args.pred_dir}")

    dataset = ShockWaveDataset(args.h5)

    print(f"Rendering {len(npz_paths)} sample(s) -> {out_dir}")
    for npz_path in npz_paths:
        sample_id = npz_path.stem
        data = np.load(npz_path)
        pred = {name: data[name] for name in CHANNEL_NAMES}
        gamma = float(data["gamma"])
        gt = load_ground_truth(dataset, sample_id)
        render_video(
            pred=pred, gt=gt, sample_id=sample_id, gamma=gamma,
            out_path=out_dir / f"{sample_id}.mp4", fps=args.fps, dpi=args.dpi,
        )

    print(f"\nDone! {len(npz_paths)} video(s) -> {out_dir}")


if __name__ == "__main__":
    main()
