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
        normalization.json         # the channel stats the latents were built with
        split_manifest.json        # which ids went to train vs val (if --eval-sims)
        val/
            latents/<sample_id>.pt # held-out split, same layout
            scalars/<sample_id>.pt

With ``--eval-sims N`` the held-out simulations are written under ``val/`` instead
of the top-level ``latents``/``scalars``, so ``PrecomputedDataset(output_dir)``
sees only the training split and ``PrecomputedDataset(output_dir/"val")`` sees
only the held-out one. Pass ``val/`` to the DiT trainer as
``validation.data_root``.

Usage:
    python scripts/preprocess_dataset.py /data/shockwave_dataset/train.h5 \
        --output-dir /data/preprocessed \
        --inflate-checkpoint outputs/vae_meaninit_wsd15_h1x2_lr1x_2gpu/checkpoints/vae_shockwave_best.safetensors \
        --eval-sims 675
"""

import json
from pathlib import Path

import torch
import typer
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from windinet.inference.model_loader import load_vae, load_inflated_vae, LtxvModelVersion
from windinet.latent_utils import encode_video
from windinet.vae_adapter import latent_space_fingerprint

from windinet.training.shockwave_data import (
    CHANNEL_NAMES,
    ShockWaveDataset,
    build_shockwave_video,
    extract_scalars,
    load_channel_normalization,
)

from windinet.utils import get_default_device, logger

console = Console()
app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


def _parse_floats(raw: str | None) -> list[float] | None:
    return None if raw is None else [float(v) for v in raw.split(",")]


def _resolve_normalization(
    inflate_checkpoint: Path,
    vae_config: str | None,
    channel_mean: str | None,
    channel_std: str | None,
    normalization_clip: float | None,
) -> tuple[list[float], list[float], float, str]:
    """Determine the channel normalization the latents must be built with.

    The VAE was finetuned on fields normalized as ``(x - mean) / (std * clip)``
    and clamped to [-1, 1]. Encoding raw physical values instead would put the
    input far outside the VAE's training distribution and produce meaningless
    latents, so the stats used here MUST be the ones the VAE saw. They are
    therefore taken from the VAE run's own ``training_config.yaml`` by default
    (found next to the checkpoint), and only overridden explicitly.

    Returns (mean, std, clip, source description).
    """
    explicit_mean = _parse_floats(channel_mean)
    explicit_std = _parse_floats(channel_std)
    if (explicit_mean is None) != (explicit_std is None):
        raise typer.BadParameter("--channel-mean and --channel-std must be given together")
    if explicit_mean is not None:
        return explicit_mean, explicit_std, normalization_clip or 5.0, "command line"

    # Auto-discover: outputs/<run>/checkpoints/vae_*.safetensors
    #             -> outputs/<run>/training_config.yaml
    config_path = Path(vae_config) if vae_config else inflate_checkpoint.parent.parent / "training_config.yaml"
    if not config_path.is_file():
        raise typer.BadParameter(
            f"Cannot determine channel normalization: no VAE training config at {config_path}. "
            "Pass --vae-config, or give --channel-mean/--channel-std explicitly. Encoding "
            "un-normalized fields would silently produce unusable latents, so there is no default."
        )
    stats = load_channel_normalization(config_path)
    clip = normalization_clip if normalization_clip is not None else stats["normalization_clip"]
    return stats["channel_mean"], stats["channel_std"], float(clip), str(config_path)


@app.command()
def main(
    data_root: str = typer.Argument(..., help="Path to the ShockWave HDF5 split (e.g. /data/shockwave/train.h5)"),
    output_dir: str = typer.Option(..., help="Output directory for preprocessed data"),
    model_source: str = typer.Option(
        default="LTXV_2B_0.9.6_DEV",
        help="LTX-Video model version for VAE encoding",
    ),
    # num_sim_frames: int = typer.Option(
    #     default=112,
    #     help="Number of simulation frames to use (112 + 1 conditioning = 113 total)",
    # ),
    device: str = typer.Option(default=str(get_default_device()), help="Device for VAE encoding"),
    max_samples: int = typer.Option(default=0, help="Limit number of samples (0 = all)"),
    inflate_checkpoint: str = typer.Option(
        default=None,
        help="Finetuned inflate-mode VAE checkpoint (.safetensors). REQUIRED to encode "
        "4-channel CFD fields: the base VAE only reads 3 channels. Its grown conv_in and "
        "finetuned decoder are loaded so latents match the VAE that will decode them.",
    ),
    scalar_names: str = typer.Option(
        default="gamma",
        help="Comma-separated scalar conditioning names to extract",
    ),
    vae_config: str = typer.Option(
        default=None,
        help="VAE training_config.yaml to read the channel normalization from. "
        "Defaults to the one next to --inflate-checkpoint.",
    ),
    channel_mean: str = typer.Option(
        default=None, help="Override: comma-separated per-channel means (with --channel-std)"
    ),
    channel_std: str = typer.Option(
        default=None, help="Override: comma-separated per-channel stds (with --channel-mean)"
    ),
    normalization_clip: float = typer.Option(
        default=None, help="Override: normalization clip (defaults to the VAE config's value)"
    ),
    eval_sims: int = typer.Option(
        default=0,
        help="Hold out N simulations as a DiT validation split, written to <output-dir>/val. "
        "0 disables the split and encodes everything into the training set. Use 675 with "
        "--split-seed 42 to reproduce the VAE run's held-out set exactly.",
    ),
    split_seed: int = typer.Option(
        default=42,
        help="Seed for the train/val partition. Must match the VAE run's seed for the two "
        "splits to be identical.",
    ),
) -> None:
    """Preprocess ShockWave HDF5 simulations into VAE latents for DiT training."""
    output_path = Path(output_dir)
    latents_dir = output_path / "latents"
    scalars_dir = output_path / "scalars"
    latents_dir.mkdir(parents=True, exist_ok=True)
    scalars_dir.mkdir(parents=True, exist_ok=True)

    val_latents_dir = output_path / "val" / "latents"
    val_scalars_dir = output_path / "val" / "scalars"

    parsed_scalar_names = [s.strip() for s in scalar_names.split(",")]

    # Load dataset
    dataset = ShockWaveDataset(data_root)
    if max_samples > 0:
        if eval_sims > 0:
            raise typer.BadParameter(
                "--max-samples truncates dataset.ids, which shifts every index the split "
                "permutation is drawn against. The resulting val set would not match the "
                "VAE run's. Use one or the other, not both."
            )
        dataset.ids = dataset.ids[:max_samples]
    logger.info(f"Found {len(dataset)} samples in {data_root}")

    # Held-out split. This mirrors VaeTrainer._setup exactly -- same sorted id
    # order (ShockWaveDataset sorts the HDF5 keys), same seed, same
    # torch.randperm, same tail slice -- so that with --eval-sims 675
    # --split-seed 42 the DiT's validation simulations are byte-for-byte the
    # ones the VAE also held out. A contiguous slice would not do: the groups
    # are ordered by gamma, so it would hand entire gamma regimes to validation
    # and turn it into an extrapolation test.
    eval_indices: set[int] = set()
    if eval_sims > 0:
        if eval_sims >= len(dataset):
            raise typer.BadParameter(
                f"--eval-sims {eval_sims} leaves no training data (dataset has {len(dataset)} sims)"
            )
        split_generator = torch.Generator().manual_seed(split_seed)
        perm = torch.randperm(len(dataset), generator=split_generator).tolist()
        eval_indices = set(perm[len(dataset) - eval_sims:])
        val_latents_dir.mkdir(parents=True, exist_ok=True)
        val_scalars_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Split (seed={split_seed}): {len(dataset) - eval_sims} train, {eval_sims} val "
            f"-> {output_path / 'val'}"
        )

    # Load VAE. The CFD fields have 4 native channels, which the base 3-channel
    # LTX VAE cannot encode, so an inflate-mode finetuned checkpoint is required.
    # Latents MUST be produced by the same VAE that will later decode them.
    if not inflate_checkpoint:
        raise typer.BadParameter(
            "--inflate-checkpoint is required: the base VAE reads only 3 channels and "
            "cannot encode the 4-channel CFD fields. Pass the finetuned inflate VAE, e.g. "
            "outputs/vae_meaninit_wsd15_h1x2_lr1x_2gpu/checkpoints/vae_shockwave_best.safetensors"
        )
    logger.info(f"Loading inflated VAE ({model_source}) + checkpoint {inflate_checkpoint}...")
    vae = load_inflated_vae(model_source, inflate_checkpoint, dtype=torch.bfloat16, device=device)
    vae = vae.to(device).eval()
    vae.requires_grad_(False)

    # The fields must be normalized exactly as during VAE finetuning before they
    # are encoded, otherwise the latents are off-distribution and unusable.
    norm_mean, norm_std, norm_clip, norm_source = _resolve_normalization(
        Path(inflate_checkpoint), vae_config, channel_mean, channel_std, normalization_clip
    )
    logger.info(
        f"Channel normalization from {norm_source}: mean={norm_mean}, std={norm_std}, clip={norm_clip}"
    )
    # Record the stats so decoding (inference) can invert them with the same
    # numbers, plus a fingerprint of the latent space these latents live in.
    # Everything downstream (DiT training, inference) checks that fingerprint:
    # decoding with a VAE whose encoder differs from this one produces physically
    # wrong fields and would otherwise fail silently.
    fingerprint = latent_space_fingerprint(
        Path(inflate_checkpoint), norm_mean, norm_std, norm_clip
    )
    logger.info(f"Latent space fingerprint: {fingerprint}")
    (output_path / "normalization.json").write_text(
        json.dumps(
            {
                "channel_names": CHANNEL_NAMES,
                "channel_mean": norm_mean,
                "channel_std": norm_std,
                "normalization_clip": norm_clip,
                "vae_checkpoint": str(Path(inflate_checkpoint).resolve()),
                "latent_fingerprint": fingerprint,
                "source": norm_source,
            },
            indent=2,
        )
    )

    processed = 0
    skipped = 0
    train_ids: list[str] = []
    val_ids: list[str] = []

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

            is_eval = idx in eval_indices
            (val_ids if is_eval else train_ids).append(sample_id)
            latent_path = (val_latents_dir if is_eval else latents_dir) / f"{sample_id}.pt"
            scalar_path = (val_scalars_dir if is_eval else scalars_dir) / f"{sample_id}.pt"

            if latent_path.exists() and scalar_path.exists():
                skipped += 1
                progress.advance(task)
                continue

            try:
                # Build conditioning video and encode
                video = build_shockwave_video(
                    sample,
                    device=torch.device(device),
                    channel_mean=norm_mean,
                    channel_std=norm_std,
                    normalization_clip=norm_clip,
                )  # [1, 4, F, H, W], normalized to [-1, 1]
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
            elif device == "xpu":
                torch.xpu.empty_cache()

    # Record the partition so the val set is auditable and can be reproduced
    # (or intersected with the VAE's) without re-deriving the permutation.
    (output_path / "split_manifest.json").write_text(
        json.dumps(
            {
                "data_root": str(Path(data_root).resolve()),
                "split_seed": split_seed,
                "eval_sims": eval_sims,
                "num_train": len(train_ids),
                "num_val": len(val_ids),
                "train_ids": sorted(train_ids),
                "val_ids": sorted(val_ids),
            },
            indent=2,
        )
    )

    logger.info(f"Done: {processed} processed, {skipped} skipped")
    logger.info(f"Latents: {latents_dir}")
    logger.info(f"Scalars: {scalars_dir}")
    if eval_sims > 0:
        logger.info(f"Held-out split ({len(val_ids)} sims): {output_path / 'val'}")
    logger.info(f"Split manifest: {output_path / 'split_manifest.json'}")


if __name__ == "__main__":
    app()
