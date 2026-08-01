"""Per-cluster defaults for VAE finetuning sbatch launchers.

Single source of truth for the settings that differ by cluster -- storage
location, dataloader worker count, target effective batch -- so each cluster's
sbatch script calls one function instead of re-deriving these values inline.
Everything else about a cluster launch script (partition, environment
activation, working directory, ...) still hard-codes to that cluster and is
not meant to be portable (see scripts/lundquist/README.md); this module is
the one shared piece, since it's plain Python imported by scripts on both
clusters rather than a script that itself needs to move between them.
"""

import os

CLUSTER_DEFAULTS = {
    "lundquist": {
        "data_root": "/local/disk/hramachandran/work/wh_work/windinet/euler_mq_dataset/128x128_ds/train.h5",
        "output_root": None,  # keep output_dir as the config's own repo-relative value
        "num_dataloader_workers": 4,
        "effective_batch": 32,
    },
    "sng_pvc": {
        "data_root": "{scratch}/windinet/euler_mq_dataset/128x128_ds/train.h5",
        "output_root": "{scratch}/windinet",
        # 8 ranks x N workers all opening train.h5 concurrently on the DSS
        # network filesystem was triggering transient open failures at N=4.
        "num_dataloader_workers": 0,
        "effective_batch": 32,
    },
}


def patch_config_for_cluster(
    cfg: dict,
    cluster: str,
    num_processes: int,
    output_suffix: str,
    effective_batch: int | None = None,
) -> dict:
    """Patch a loaded VAE trainer config dict in place for `cluster`.

    Overrides data_root and num_dataloader_workers from CLUSTER_DEFAULTS,
    derives gradient_accumulation_steps from the target effective batch
    (CLUSTER_DEFAULTS[cluster]["effective_batch"], or `effective_batch` to
    override it -- e.g. lundquist's 6-GPU script targets 30, not 32, because
    32 isn't divisible by 6 ranks) and num_processes, so every cluster/GPU
    count combination trains on the same effective batch without hand-tuning
    the multiplier. output_dir gets output_suffix appended (so runs from
    different clusters/GPU counts never collide) and, if the cluster has an
    output_root, gets relocated under it (e.g. sng_pvc's $SCRATCH).
    """
    defaults = CLUSTER_DEFAULTS[cluster]
    scratch = os.environ.get("SCRATCH", "")

    cfg["data"]["data_root"] = defaults["data_root"].format(scratch=scratch)
    cfg["data"]["num_dataloader_workers"] = defaults["num_dataloader_workers"]

    batch_size = cfg["optimization"]["batch_size"]
    target_batch = defaults["effective_batch"] if effective_batch is None else effective_batch
    if target_batch % (batch_size * num_processes) != 0:
        raise ValueError(
            f"effective_batch {target_batch} not divisible by "
            f"batch_size {batch_size} x num_processes {num_processes}"
        )
    cfg["optimization"]["gradient_accumulation_steps"] = target_batch // (batch_size * num_processes)

    output_dir = cfg["output_dir"].rstrip("/") + output_suffix
    output_root = defaults["output_root"]
    if output_root is not None:
        output_dir = os.path.join(output_root.format(scratch=scratch), output_dir)
    cfg["output_dir"] = output_dir

    return cfg
