"""Per-cluster defaults for VAE finetuning sbatch launchers.

Single source of truth for the settings that differ by cluster -- storage
location, dataloader worker count, target effective batch -- so each cluster's
sbatch script calls one function instead of re-deriving these values inline.
Everything else about a cluster launch script (partition, environment
activation, working directory, ...) still hard-codes to that cluster and is
not meant to be portable (see jobs/lundquist/README.md); this module is
the one shared piece, since it's plain Python imported by scripts on both
clusters rather than a script that itself needs to move between them.
"""

import os

CLUSTER_DEFAULTS = {
    "lundquist": {
        "data_root": "/local/disk/hramachandran/work/wh_work/windinet/euler_mq_dataset/128x128_ds/train.h5",
        # Repo-relative, shared by every lundquist GPU-count variant on
        # purpose: there's one lundquist result at a time, not one per GPU
        # count, so the 2/4-GPU scripts all land here regardless of
        # output_suffix (a script that needs several arms to coexist -- e.g.
        # the now-retired batch_size sweep, EXPERIMENTS.md Open Question 23
        # -- is the exception, and always passes a real output_suffix).
        # clean_output_dir wipes it at the start of each run.
        "output_root": "finetune_vae_outputs/lundquist",
        "num_dataloader_workers": 4,
        "effective_batch": 32,
    },
    "sng_pvc": {
        "data_root": "{scratch}/windinet/euler_mq_dataset/128x128_ds/train.h5",
        # Deliberately still the old (pre-rename) name, unlike the git-tracked
        # mirror under finetune_vae_outputs/sng_pvc/ -- every run before and
        # after the 2026-08-12 directory rename lives here on $SCRATCH, so
        # keeping this constant avoids splitting checkpoints across two
        # differently-named scratch trees. jobs/sng_pvc/finetune_vae*.sbatch's
        # LOCAL_DIR mirror target is a separate hardcoded string, not derived
        # from this value, so it can (and does) use the new name regardless.
        "output_root": "{scratch}/windinet/finetune_vae_outputs_sng_pvc",
        # workers=0 -> 2 cuts wall-clock/epoch ~21% (confirmed safe at 8 ranks,
        # jobs 520301 vs 520300, see EXPERIMENTS.md "sng_pvc throughput
        # diagnostic"). workers=2 -> 4 cuts a further ~6.6% (jobs 520456 vs
        # 520301, mostly from eval, not train) with no crash; workers=8
        # (520457) gained nothing more past 4. Both 4/8 readings were only
        # 2-epoch diagnostics, not a full 15-epoch run, when this was last
        # promoted from 2 -- see EXPERIMENTS.md for that caveat if throughput
        # regresses on a full run and this needs revisiting.
        "num_dataloader_workers": 4,
        "effective_batch": 32,
    },
    "lrz_ai": {
        # Paths as seen inside the enroot container: --container-mounts binds
        # $DATA_DIR/$RESULT_DIR (set by the .job script) to /mnt/data and
        # /mnt/result respectively.
        "data_root": "/mnt/data/train.h5",
        "output_root": "/mnt/result",
        "num_dataloader_workers": 4,
        "effective_batch": 32,
    },
}


def patch_config_for_cluster(
    cfg: dict,
    cluster: str,
    num_processes: int,
    output_suffix: str,
    effective_batch: int | None = None,
    data_root: str | None = None,
) -> dict:
    """Patch a loaded VAE trainer config dict in place for `cluster`.

    Overrides data_root and num_dataloader_workers from CLUSTER_DEFAULTS,
    derives gradient_accumulation_steps from the target effective batch
    (CLUSTER_DEFAULTS[cluster]["effective_batch"], or `effective_batch` to
    override it -- e.g. lundquist's debug launcher passes effective_batch=1
    to hit an exact one-optimizer-step-per-sim diagnostic instead of the
    cluster default) and num_processes, so every cluster/GPU count
    combination trains on the same effective batch without hand-tuning
    the multiplier. output_dir gets output_suffix appended, then gets
    relocated under the cluster's output_root (its own top-level results
    folder, e.g. sng_pvc's $SCRATCH). output_suffix is typically "" for
    scripts that intentionally share one cluster-wide output_dir (lundquist's
    2/4-GPU variants all overwrite the same run on purpose -- see
    CLUSTER_DEFAULTS); pass a real suffix only when two scripts sharing a
    cluster must NOT overwrite each other's output (e.g. lundquist's
    batch_size sweep, one output_dir per `BATCH_SIZE`).

    data_root overrides CLUSTER_DEFAULTS[cluster]["data_root"] verbatim (no
    {scratch} formatting -- pass an already-expanded path) for runs against a
    dataset other than the cluster's default resolution, e.g. 256x256_ds
    instead of the default 128x128_ds. Leave unset to keep prior behavior.
    """
    defaults = CLUSTER_DEFAULTS[cluster]
    scratch = os.environ.get("SCRATCH", "")

    cfg["data"]["data_root"] = data_root if data_root is not None else defaults["data_root"].format(scratch=scratch)
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
