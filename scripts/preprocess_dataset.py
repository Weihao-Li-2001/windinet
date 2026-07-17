# checked and changed

"""
Preprocess ShockWave simulations into latent representations for diffusion transformer training.

Loads raw CFD simulations from an HDF5 dataset, encodes each sample through the
LTX-Video VAE, and saves latent tensors together with scalar conditioning values
as `.pt` files.

Expected input dataset::

    data_root/
        train.h5 (or val.h5 / test.h5)
            <sample_id>/
                density        [T, H, W]
                momentum_x     [T, H, W]
                momentum_y     [T, H, W]
                pressure       [T, H, W]
                metadata such as gamma

Output structure (compatible with PrecomputedDataset)::

    output_dir/
        latents/<sample_id>.pt     # {latents, num_frames, height, width}
        scalars/<sample_id>.pt     # {scalars, scalar_names}

Usage:
    python scripts/preprocess_dataset.py /data/shockwave_dataset/train.h5 \
        --output-dir /data/preprocessed
"""

from pathlib import Path

import torch
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from windinet.inference.model_loader import load_vae, LtxvModelVersion
from windinet.latent_utils import encode_video

from windinet.training.shockwave_data import (
    ShockWaveDataset,
    build_shockwave_video,
    extract_scalars,
)

from windinet.utils import logger

console = Console()
app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


@app.command()
def main(
    data_root: str = typer.Argument(..., help="Path to dataset split directory (e.g. /data/wind_dataset/train)"),
    output_dir: str = typer.Option(..., help="Output directory for preprocessed data"),
    model_source: str = typer.Option(
        default="LTXV_2B_0.9.6_DEV",
        help="LTX-Video model version for VAE encoding",
    ),
    # num_sim_frames: int = typer.Option(
    #     default=112,
    #     help="Number of simulation frames to use (112 + 1 conditioning = 113 total)",
    # ),
    device: str = typer.Option(default="cuda", help="Device for VAE encoding"),
    max_samples: int = typer.Option(default=0, help="Limit number of samples (0 = all)"),
    scalar_names: str = typer.Option(
        default="gamma",
        help="Comma-separated scalar conditioning names to extract",
    ),
) -> None:
    """Preprocess wind field NPZ data into VAE latents for DiT training."""
    output_path = Path(output_dir)
    latents_dir = output_path / "latents"
    scalars_dir = output_path / "scalars"
    latents_dir.mkdir(parents=True, exist_ok=True)
    scalars_dir.mkdir(parents=True, exist_ok=True)

    parsed_scalar_names = [s.strip() for s in scalar_names.split(",")]

    # Load dataset
    dataset = ShockWaveDataset(data_root)
    if max_samples > 0:
        dataset.ids = dataset.ids[:max_samples]
    logger.info(f"Found {len(dataset)} samples in {data_root}")

    # Load VAE
    logger.info(f"Loading VAE ({model_source})...")
    vae = load_vae(model_source, dtype=torch.bfloat16)
    vae = vae.to(device).eval()
    vae.requires_grad_(False)

    processed = 0
    skipped = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Encoding", total=len(dataset))

        for idx in range(len(dataset)):
            sample = dataset[idx]
            sample_id = sample["id"]

            latent_path = latents_dir / f"{sample_id}.pt"
            scalar_path = scalars_dir / f"{sample_id}.pt"

            if latent_path.exists() and scalar_path.exists():
                skipped += 1
                progress.advance(task)
                continue

            try:
                # Build conditioning video and encode
                video = build_shockwave_video(
                    sample,
                    device=torch.device(device)
                )  # [1, 4, F, H, W]
                video = video.permute(0, 2, 1, 3, 4)  # [1, F, C, H, W] for encode_video

                with torch.no_grad():
                    latent_dict = encode_video(vae, video, device=torch.device(device), dtype=torch.bfloat16)

                # Save latents (squeeze batch dim so DataLoader collation gives [B, seq, C])
                torch.save({
                    "latents": latent_dict["latents"].squeeze(0).cpu(),
                    "num_frames": latent_dict["num_frames"],
                    "height": latent_dict["height"],
                    "width": latent_dict["width"],
                }, latent_path)

                # Save scalars
                scalars = extract_scalars(sample["meta"], parsed_scalar_names)
                scalar_tensor = torch.tensor(
                    [scalars[name] for name in parsed_scalar_names],
                    dtype=torch.float32,
                )
                torch.save({
                    "scalars": scalar_tensor,
                    "scalar_names": parsed_scalar_names,
                }, scalar_path)

                processed += 1

            except Exception as e:
                logger.warning(f"Failed to process {sample_id}: {e}")
                skipped += 1

            progress.advance(task)

            if device == "cuda":
                torch.cuda.empty_cache()

    logger.info(f"Done: {processed} processed, {skipped} skipped")
    logger.info(f"Latents: {latents_dir}")
    logger.info(f"Scalars: {scalars_dir}")


if __name__ == "__main__":
    app()
