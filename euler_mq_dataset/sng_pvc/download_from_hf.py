"""
Download one resolution's worth of the euler_mq ShockWave CFD dataset from
HuggingFace (rha6696/euler_mq) to $SCRATCH/windinet/euler_mq_dataset/.

The repo's dataset card states the native simulation resolution is 512x512,
downsampled to 256x256 and 128x128 -- all three are separate top-level
folders in the repo (128x128_ds, 256x256_ds, 512x512_orig), only one of
which (128x128_ds) has been used by this project so far.

Usage:
    python euler_mq_dataset/sng_pvc/download_from_hf.py                 # 128x128_ds (default, previous hardcoded behavior)
    python euler_mq_dataset/sng_pvc/download_from_hf.py 256x256_ds
    python euler_mq_dataset/sng_pvc/download_from_hf.py 512x512_orig
"""

import os

import typer
from huggingface_hub import snapshot_download

RESOLUTIONS = ["128x128_ds", "256x256_ds", "512x512_orig"]


def main(
    resolution: str = typer.Argument(
        "128x128_ds",
        help=f"Which resolution folder to download. One of: {', '.join(RESOLUTIONS)}.",
    ),
) -> None:
    if resolution not in RESOLUTIONS:
        raise typer.BadParameter(f"resolution must be one of {RESOLUTIONS}, got {resolution!r}")

    local_dir = os.path.join(os.environ["SCRATCH"], "windinet", "euler_mq_dataset")
    os.makedirs(local_dir, exist_ok=True)

    snapshot_download(
        repo_id="rha6696/euler_mq",
        repo_type="dataset",
        allow_patterns=f"{resolution}/*",
        local_dir=local_dir,
    )


if __name__ == "__main__":
    typer.run(main)
