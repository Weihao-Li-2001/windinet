"""
Download one resolution's worth of the euler_mq ShockWave CFD dataset from
HuggingFace (rha6696/euler_mq) onto lrz_ai's DSS dataset storage.

Same three top-level resolution folders as euler_mq_dataset/sng_pvc/
download_from_hf.py (see that script's docstring for the dataset-card
detail: native sim resolution is 512x512, 512x512_orig is that native
resolution -- NOT downsampled, despite the naming symmetry with the other
two -- 256x256_ds and 128x128_ds are downsampled from it).

Downloads next to the existing 128x128_ds (jobs/lrz_ai/finetune_vae_1gpu.job's
$DATA_DIR points at .../Euler_MQ/data/128x128_ds), so 256x256_ds/512x512_orig
land as sibling folders under the same .../Euler_MQ/data/ directory --
matches sng_pvc's layout ($SCRATCH/windinet/euler_mq_dataset/<resolution>/).

Needs outbound internet access to huggingface.co -- run this from a login
node, not inside a training job (jobs/lrz_ai/finetune_vae_1gpu.job's
container has no confirmed internet access, and every training job in this
repo sets HF_HUB_OFFLINE=1 during the run).

Usage (run inside the `windinet` conda env, which has huggingface_hub):
    python euler_mq_dataset/lrz_ai/download_from_hf.py 256x256_ds
    python euler_mq_dataset/lrz_ai/download_from_hf.py 512x512_orig
"""

import typer
from huggingface_hub import snapshot_download

RESOLUTIONS = ["128x128_ds", "256x256_ds", "512x512_orig"]

# Parent of $DATA_DIR in jobs/lrz_ai/finetune_vae_1gpu.job -- change this if
# that script's $DATA_DIR ever moves.
DATA_ROOT = "/dss/dssfs02/pn82ku/pn82ku-dss-0000/neptuna_stuff/datasets/Euler_MQ/data"


def main(
    resolution: str = typer.Argument(
        ...,
        help=f"Which resolution folder to download. One of: {', '.join(RESOLUTIONS)}.",
    ),
) -> None:
    if resolution not in RESOLUTIONS:
        raise typer.BadParameter(f"resolution must be one of {RESOLUTIONS}, got {resolution!r}")

    snapshot_download(
        repo_id="rha6696/euler_mq",
        repo_type="dataset",
        allow_patterns=f"{resolution}/*",
        local_dir=DATA_ROOT,
    )


if __name__ == "__main__":
    typer.run(main)
