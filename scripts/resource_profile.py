#!/usr/bin/env python
"""Compute the compute-resource figures a cluster/allocation proposal asks for.

Answers the six standard proposal questions for a WinDiNet production run:

  1. Average number of processes
  2. Average job memory (total usage over all nodes, GB)
  3. Maximum amount of memory per process (MB)
  4. Total amount of data to transfer to/from (GB)
  5. Frequency and size of data output and input
  6. Number of files and size of each file in a typical production run

Everything that can be *counted* (dataset inventory, file sizes, checkpoint
cadence, per-epoch read volume) is read straight off disk and out of the
launcher/config files -- no estimates. The two figures that genuinely need a
measurement (host RSS and device memory per rank) come from `--probe`, which
runs the real trainers for a few steps under a process-tree memory sampler
rather than from an analytic guess.

Usage::

    # inventory only (seconds, no GPU needed)
    python scripts/resource_profile.py

    # + measured memory: runs the real VAE trainer on 2 sims for 1 epoch
    python scripts/resource_profile.py --probe vae --probe-processes 2

    # everything, including the DiT (needs ~35 GB of device memory)
    python scripts/resource_profile.py --probe all --json profile.json

The `--probe dit` / `--probe all` paths chain off the VAE probe: the short VAE
run writes a real checkpoint, that checkpoint encodes a few sims into real
latents, and the DiT trainer then runs on those. That way the latent size and
both checkpoint sizes are measured artifacts, not derived numbers.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

console = Console(width=150)
app = typer.Typer(pretty_exceptions_enable=False, add_completion=False)

GB = 1024**3
MB = 1024**2

# Weights pulled once per machine (not per job) into the HuggingFace hub cache:
# LTXV_2B_0.9.6_DEV's single safetensors blob plus the 0.9.5 VAE/transformer
# configs the loader reads alongside it. Measured from pretrained/hub when it
# exists, this constant is only the fallback for a machine without one.
LTXV_WEIGHTS_FALLBACK_BYTES = 9 * GB


# ----------------------------------------------------------------------------
# 1. Dataset inventory
# ----------------------------------------------------------------------------


@dataclass
class DatasetProfile:
    path: Path
    n_sims: int
    fields: list[str]
    frames: int
    height: int
    width: int
    dtype: str
    itemsize: int
    gammas: list[float]
    backing_files: dict[str, int]  # resolved path -> size in bytes
    index_file_bytes: int
    logical_bytes_per_sim: int
    stored_bytes_per_sim: int

    @property
    def logical_bytes(self) -> int:
        return self.logical_bytes_per_sim * self.n_sims

    @property
    def stored_bytes(self) -> int:
        return sum(self.backing_files.values()) + self.index_file_bytes

    @property
    def compression_ratio(self) -> float:
        return self.logical_bytes / self.stored_bytes if self.stored_bytes else 0.0


def scan_dataset(h5_path: Path, sample_groups: int = 8) -> DatasetProfile:
    """Inventory an Euler-MQ `train.h5`.

    `train.h5` is an index of HDF5 external links, not a data file: each
    `<sample_id>` group points into one of the per-gamma `train_subsets/*.hdf5`
    shards. `h5py` follows those transparently, so the group *contents* read
    normally while the index itself is only a few hundred kB -- the bytes that
    actually move over the network live in the shards, and those are what this
    resolves and sizes.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        ids = list(f.keys())
        backing: dict[str, int] = {}
        for sid in ids:
            link = f.get(sid, getlink=True)
            fname = getattr(link, "filename", None)
            if fname is None:
                continue  # data lives in this file itself, not an external shard
            resolved = str((h5_path.parent / fname).resolve())
            backing.setdefault(resolved, Path(resolved).stat().st_size)

        first = f[ids[0]]
        fields = sorted(first.keys())
        shape = first[fields[0]].shape  # (T, 1, H, W)
        dtype = first[fields[0]].dtype
        frames, _, height, width = shape

        # Storage size of the first few groups, averaged: HDF5 reports the
        # post-gzip byte count per dataset, which is the real read volume.
        stored, logical = 0, 0
        n = min(sample_groups, len(ids))
        for sid in ids[:n]:
            for fld in fields:
                d = f[sid][fld]
                stored += d.id.get_storage_size()
                logical += d.nbytes
        stored_per_sim = stored // n
        logical_per_sim = logical // n

        gammas = sorted({float(f[sid].attrs["gamma"]) for sid in ids[:: max(1, len(ids) // 200)]})

    return DatasetProfile(
        path=h5_path,
        n_sims=len(ids),
        fields=fields,
        frames=frames,
        height=height,
        width=width,
        dtype=str(dtype),
        itemsize=dtype.itemsize,
        gammas=gammas,
        backing_files=backing,
        index_file_bytes=h5_path.stat().st_size,
        logical_bytes_per_sim=logical_per_sim,
        stored_bytes_per_sim=stored_per_sim,
    )


# ----------------------------------------------------------------------------
# 1b. Synthetic higher-resolution datasets, for probing 256/512 without the data
# ----------------------------------------------------------------------------

# The 256x256_ds / 512x512_orig splits live only on the clusters (and on the
# HuggingFace dataset repo), but device and host memory are set by tensor
# *shape*, not by the values in it. Upsampling a handful of real 128x128 sims
# to the target resolution reproduces the exact shapes the trainer would see,
# so the resulting peak-memory readings are measurements of the real code path
# rather than an analytic scaling of the 128 number. Only the dataset-size and
# throughput figures still have to be scaled, and those scale exactly with
# pixel count.
RESOLUTION_CONFIGS = {
    128: "configs/finetune_vae/finetune_vae_whole_structure_baseline_ep30.yaml",
    256: "configs/finetune_vae/finetune_vae_whole_structure_baseline_ep30_256res.yaml",
    512: "configs/finetune_vae/finetune_vae_whole_structure_baseline_512res.yaml",
}

# Real per-resolution dataset sizes, as published on the HuggingFace dataset
# repo (`rha6696/euler_mq`). Filled in live by `fetch_hf_dataset_sizes()` when
# the machine has outbound access; these are the recorded fallback so an
# offline run still reports true sizes rather than a guess.
HF_TRAIN_BYTES = {
    128: int(82.02 * GB),
    256: int(319.96 * GB),
    512: int(1259.20 * GB),
}
HF_DATASET_REPO = "rha6696/euler_mq"
HF_RESOLUTION_DIRS = {128: "128x128_ds", 256: "256x256_ds", 512: "512x512_orig"}


def fetch_hf_dataset_sizes(timeout: float = 60.0) -> dict[int, dict[str, int]] | None:
    """Real per-resolution train/test sizes from the HuggingFace dataset repo.

    Sizes come from the repo tree listing -- no download. Returns None if the
    machine has no outbound access, in which case callers fall back to
    HF_TRAIN_BYTES.
    """
    try:
        from huggingface_hub import HfApi

        entries = list(
            HfApi().list_repo_tree(HF_DATASET_REPO, repo_type="dataset", recursive=True)
        )
    except Exception:
        return None

    out: dict[int, dict[str, int]] = {}
    for res, prefix in HF_RESOLUTION_DIRS.items():
        train = test = 0
        for e in entries:
            size = getattr(e, "size", None)
            if not size or not e.path.startswith(prefix + "/"):
                continue
            if "/train_subsets/" in e.path:
                train += size
            elif "/test_subsets/" in e.path:
                test += size
        if train:
            out[res] = {"train_bytes": train, "test_bytes": test}
    return out or None


def synthesize_resolution(src_h5: Path, out_h5: Path, resolution: int, n_sims: int) -> Path:
    """Write a structurally-identical HDF5 at `resolution`, upsampled from `src_h5`.

    Same group/dataset layout, dtype, gzip filter and chunk *shape scaling* as
    the real shards, so the dataloader and the trainer take the identical code
    path. Values are bilinearly upsampled real fields, which keeps them in the
    physical range the normalization expects -- important only so the run does
    not produce NaNs and bail before reaching peak memory.
    """
    import h5py
    import torch
    import torch.nn.functional as F

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src_h5, "r") as src, h5py.File(out_h5, "w") as dst:
        ids = list(src.keys())[:n_sims]
        for sid in ids:
            group = dst.create_group(sid)
            group.attrs["gamma"] = src[sid].attrs["gamma"]
            for field in src[sid]:
                arr = src[sid][field][:]  # (T, 1, H, W)
                t = torch.from_numpy(arr).float()
                up = F.interpolate(
                    t.squeeze(1).unsqueeze(1),
                    size=(resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                )
                group.create_dataset(
                    field,
                    data=up.numpy().astype(arr.dtype),
                    chunks=(13, 1, resolution // 8, resolution // 4),
                    compression="gzip",
                )
    return out_h5


# ----------------------------------------------------------------------------
# 2. Job shape, read from the launchers + configs rather than hard-coded
# ----------------------------------------------------------------------------


@dataclass
class JobShape:
    """One production job's process/parallelism layout."""

    name: str
    cluster: str
    launcher: str
    nodes: int
    ranks_per_node: int
    cpus_per_task: int | None
    dataloader_workers: int
    batch_size: int
    grad_accum: int
    wall_limit: str

    @property
    def ranks(self) -> int:
        return self.nodes * self.ranks_per_node

    @property
    def os_processes(self) -> int:
        """Ranks + their persistent dataloader worker processes.

        The VAE trainer builds three loaders per rank (train / eval / vis) but
        only one is iterated at a time, so worker processes are `workers` per
        rank concurrently, not 3x that.
        """
        return self.ranks * (1 + self.dataloader_workers)

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum * self.ranks


def _sbatch_int(text: str, key: str) -> int | None:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#SBATCH") and key in line:
            tail = line.split(key, 1)[1].lstrip("= ").split()[0].split("#")[0]
            digits = "".join(c for c in tail if c.isdigit())
            if digits:
                return int(digits)
    return None


def _sbatch_str(text: str, key: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#SBATCH") and key in line:
            return line.split(key, 1)[1].lstrip("= ").split()[0]
    return None


def read_job_shapes() -> list[JobShape]:
    """Parse the checked-in Slurm launchers into JobShape rows.

    Reads the real `#SBATCH` headers and the real per-cluster defaults from
    `windinet.cluster_config`, so this stays correct when a launcher changes
    instead of restating numbers that drift.
    """
    from windinet.cluster_config import CLUSTER_DEFAULTS

    shapes: list[JobShape] = []

    specs = [
        # (name, cluster, launcher path, ranks-per-node source, batch_size)
        ("VAE finetune (production)", "sng_pvc", "jobs/sng_pvc/finetune_vae.sbatch", "NUM_XPUS", 4),
        ("DiT training (production)", "sng_pvc", "jobs/sng_pvc/train_dit.sbatch", "NUM_XPUS", None),
        ("VAE finetune (H100, 2 GPU)", "lrz_ai", "jobs/lrz_ai/finetune_vae_2gpu.job", "gres", 16),
        ("VAE finetune (H100, 4 GPU)", "lrz_ai", "jobs/lrz_ai/finetune_vae_4gpu.job", "gres", 1),
    ]

    dit_cfg = yaml.safe_load((REPO_ROOT / "configs/dit/train_dit.yaml").read_text())

    for name, cluster, rel, rank_src, batch_size in specs:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        text = path.read_text()
        nodes = _sbatch_int(text, "--nodes") or 1

        if rank_src == "gres":
            gres = _sbatch_str(text, "--gres") or "gpu:1"
            ranks_per_node = int(gres.rsplit(":", 1)[-1])
        else:
            # `export NUM_XPUS=8` in the body, not an #SBATCH header.
            ranks_per_node = 1
            for line in text.splitlines():
                if line.strip().startswith("export NUM_XPUS="):
                    ranks_per_node = int(line.split("=", 1)[1].split()[0])

        is_dit = "train_dit" in rel
        if is_dit:
            batch = dit_cfg["optimization"]["batch_size"]
            accum = dit_cfg["optimization"]["gradient_accumulation_steps"]
            workers = dit_cfg["data"]["num_dataloader_workers"]
        else:
            defaults = CLUSTER_DEFAULTS[cluster]
            batch = batch_size
            workers = defaults["num_dataloader_workers"]
            # patch_config_for_cluster derives accum from the target effective batch.
            accum = defaults["effective_batch"] // (batch * nodes * ranks_per_node)

        shapes.append(
            JobShape(
                name=name,
                cluster=cluster,
                launcher=rel,
                nodes=nodes,
                ranks_per_node=ranks_per_node,
                cpus_per_task=_sbatch_int(text, "--cpus-per-task"),
                dataloader_workers=workers,
                batch_size=batch,
                grad_accum=accum,
                wall_limit=_sbatch_str(text, "--time") or _sbatch_str(text, "-t") or "?",
            )
        )
    return shapes


# ----------------------------------------------------------------------------
# 3. Output inventory: what a finished run leaves on disk
# ----------------------------------------------------------------------------


@dataclass
class OutputItem:
    kind: str
    per_run_count: int
    bytes_each: int
    cadence: str
    note: str = ""

    @property
    def total_bytes(self) -> int:
        return self.per_run_count * self.bytes_each


def measured_png_bytes() -> int | None:
    """Median size of a reconstruction panel from the mirrored runs in-repo."""
    pngs = sorted((REPO_ROOT / "finetune_vae_outputs").rglob("visualizations/**/frame_*.png"))
    if not pngs:
        return None
    sizes = sorted(p.stat().st_size for p in pngs)
    return sizes[len(sizes) // 2]


def vae_output_inventory(
    cfg: dict,
    ckpt_bytes: int | None,
    state_bytes: int | None,
    png_bytes: int | None,
) -> list[OutputItem]:
    """File-by-file output plan for one VAE finetuning run, from its config."""
    epochs = cfg["optimization"]["epochs"]
    vis = cfg["visualization"]
    ck = cfg["checkpoints"]
    items: list[OutputItem] = []

    n_vis_epochs = epochs // max(1, vis.get("interval_epochs", 1)) if vis.get("enabled") else 0
    n_panels = n_vis_epochs * vis["num_samples"] * len(vis["frame_numbers"])
    if n_panels:
        items.append(
            OutputItem(
                "visualizations/epoch_####/<sim>/frame_####.png",
                n_panels,
                png_bytes or 550_000,
                f"{vis['num_samples']} sims x {len(vis['frame_numbers'])} frames every "
                f"{vis['interval_epochs']} epoch(s)",
                "rank 0 only",
            )
        )

    # save_best_only keeps two fixed weight slots plus any pinned save_epochs,
    # rather than one file per epoch; keep_last_n bounds the rolling state files.
    if ck.get("save_best_only", True):
        n_weight_files = 2 + len(ck.get("save_epochs", []))
        writes = f"rewritten every {ck['interval']} epoch(s) (best slot only on improvement)"
    else:
        n_weight_files = min(epochs, ck.get("keep_last_n", 1) if ck.get("keep_last_n", 1) > 0 else epochs)
        writes = f"one per {ck['interval']} epoch(s), pruned to keep_last_n={ck.get('keep_last_n')}"

    if ckpt_bytes:
        items.append(OutputItem("checkpoints/vae_shockwave_*.safetensors", n_weight_files, ckpt_bytes, writes))
    if state_bytes and ck.get("save_last_state", True):
        items.append(
            OutputItem(
                "checkpoints/vae_shockwave_*.state.pt",
                max(1, ck.get("keep_last_n", 1)),
                state_bytes,
                writes,
                "optimizer + scheduler + RNG, for resume_from",
            )
        )

    items.append(OutputItem("metrics/metrics.csv", 1, 300 * epochs, "appended once per epoch"))
    items.append(OutputItem("metrics/loss_curves.png", 1, 120_000, "rewritten once per epoch"))
    items.append(OutputItem("training_config.yaml", 1, 4_000, "once at startup"))
    items.append(OutputItem("slurm .out/.err logs", 2, 200_000, "streamed"))
    return items


def dit_output_inventory(cfg: dict, dit_ckpt_bytes: int | None, scalar_ckpt_bytes: int | None) -> list[OutputItem]:
    steps = cfg["optimization"]["steps"]
    ck = cfg["checkpoints"]
    interval = ck["interval"]
    keep = ck["keep_last_n"]
    n_written = steps // interval
    n_retained = n_written if keep < 0 else min(keep, n_written)
    items = [
        OutputItem(
            "checkpoints/model_weights_step_#####.safetensors",
            n_retained,
            dit_ckpt_bytes or 0,
            f"written every {interval} optimizer steps ({n_written} writes over {steps} steps), "
            f"pruned to keep_last_n={keep}",
            f"{n_written} writes total -- {(n_written * (dit_ckpt_bytes or 0)) / GB:.0f} GB of "
            "write traffic even though only the retained window persists",
        ),
        OutputItem(
            "checkpoints/scalar_embedding_step_#####.safetensors",
            n_retained,
            scalar_ckpt_bytes or 0,
            f"alongside each transformer checkpoint",
        ),
        OutputItem("latent_provenance.json", 1, 1_000, "once at startup"),
        OutputItem("epoch_log.json", 1, 200 * n_written, "appended per logged epoch"),
        OutputItem("slurm .out/.err logs", 2, 400_000, "streamed"),
    ]
    return items


# ----------------------------------------------------------------------------
# 4. Memory probe: process-tree sampler + real short training runs
# ----------------------------------------------------------------------------


def _descendants(root_pid: int) -> list[int]:
    """PIDs of `root_pid` and everything below it, via /proc."""
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            stat = Path("/proc", entry, "stat").read_text()
            # comm can contain spaces/parens; ppid is the field after the last ')'
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(entry))

    out, stack = [], [root_pid]
    while stack:
        pid = stack.pop()
        out.append(pid)
        stack.extend(children.get(pid, []))
    return out


def _mem_bytes(pid: int) -> tuple[int, int, int]:
    """(RSS, PSS, USS) for `pid`, in bytes.

    RSS counts every resident page, including file-backed pages shared with
    other ranks. safetensors loads weights via mmap, so N ranks reading the
    same checkpoint each report the full file in RSS even though it occupies
    physical memory once -- summing RSS across ranks double-counts, and even a
    single rank's RSS overstates what it privately needs.

    PSS divides each shared page by the number of processes mapping it, so
    summing PSS across the tree gives true physical usage. USS is private
    pages only -- what would actually be freed if the process exited, and the
    honest answer to "maximum memory per process".
    """
    try:
        rollup = Path("/proc", str(pid), "smaps_rollup").read_text()
    except OSError:
        # smaps_rollup needs kernel >= 4.14 and matching ownership; fall back
        # to RSS alone rather than losing the sample entirely.
        try:
            for line in Path("/proc", str(pid), "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    v = int(line.split()[1]) * 1024
                    return v, v, v
        except (OSError, ValueError, IndexError):
            pass
        return 0, 0, 0

    vals = {}
    for line in rollup.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith(":"):
            try:
                vals[parts[0][:-1]] = int(parts[1]) * 1024
            except ValueError:
                continue
    rss = vals.get("Rss", 0)
    pss = vals.get("Pss", rss)
    uss = vals.get("Private_Clean", 0) + vals.get("Private_Dirty", 0)
    return rss, pss, uss


def _rss_bytes(pid: int) -> int:
    return _mem_bytes(pid)[0]


def _gpu_process_memory() -> dict[int, int]:
    """pid -> device bytes, from nvidia-smi. Empty on a non-NVIDIA machine."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    result: dict[int, int] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            result[int(parts[0])] = int(parts[1]) * MB
    return result


class TreeSampler(threading.Thread):
    """Samples host RSS and device memory over a process tree while it runs.

    Peak *sum* over the tree answers "job memory"; peak *max over single
    processes* answers "memory per process". Both have to come from the same
    sampler -- summing independently-measured per-process high-water marks
    would over-count, since ranks do not peak simultaneously.
    """

    def __init__(self, root_pid: int, interval: float = 0.25) -> None:
        super().__init__(daemon=True)
        self.root_pid = root_pid
        self.interval = interval
        self._stop_event = threading.Event()
        self.peak_tree_rss = 0
        self.peak_proc_rss = 0
        self.peak_tree_pss = 0
        self.peak_proc_pss = 0
        self.peak_proc_uss = 0
        self.peak_tree_gpu = 0
        self.peak_proc_gpu = 0
        self.peak_process_count = 0
        self.samples = 0
        self._rss_sum_acc = 0
        self._proc_count_acc = 0

    def run(self) -> None:
        while not self._stop_event.is_set():
            pids = _descendants(self.root_pid)
            mem = {pid: _mem_bytes(pid) for pid in pids}
            live = {pid: t for pid, t in mem.items() if t[0]}
            if live:
                total = sum(t[0] for t in live.values())
                self.peak_tree_rss = max(self.peak_tree_rss, total)
                self.peak_proc_rss = max(self.peak_proc_rss, max(t[0] for t in live.values()))
                self.peak_tree_pss = max(self.peak_tree_pss, sum(t[1] for t in live.values()))
                self.peak_proc_pss = max(self.peak_proc_pss, max(t[1] for t in live.values()))
                self.peak_proc_uss = max(self.peak_proc_uss, max(t[2] for t in live.values()))
                self.peak_process_count = max(self.peak_process_count, len(live))
                self._rss_sum_acc += total
                self._proc_count_acc += len(live)
                self.samples += 1

            gpu = _gpu_process_memory()
            mine = {pid: b for pid, b in gpu.items() if pid in mem}
            if mine:
                self.peak_tree_gpu = max(self.peak_tree_gpu, sum(mine.values()))
                self.peak_proc_gpu = max(self.peak_proc_gpu, max(mine.values()))

            self._stop_event.wait(self.interval)

    def stop(self) -> dict:
        self._stop_event.set()
        self.join(timeout=5)
        return {
            "peak_tree_rss_bytes": self.peak_tree_rss,
            "peak_process_rss_bytes": self.peak_proc_rss,
            "peak_tree_pss_bytes": self.peak_tree_pss,
            "peak_process_pss_bytes": self.peak_proc_pss,
            "peak_process_uss_bytes": self.peak_proc_uss,
            "peak_tree_gpu_bytes": self.peak_tree_gpu,
            "peak_process_gpu_bytes": self.peak_proc_gpu,
            "peak_process_count": self.peak_process_count,
            "mean_tree_rss_bytes": int(self._rss_sum_acc / self.samples) if self.samples else 0,
            "mean_process_count": round(self._proc_count_acc / self.samples, 1) if self.samples else 0,
            "samples": self.samples,
        }


def run_sampled(cmd: list[str], cwd: Path, env: dict, log_path: Path) -> tuple[int, dict]:
    """Run `cmd`, sampling its whole process tree until it exits."""
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        sampler = TreeSampler(proc.pid)
        sampler.start()
        rc = proc.wait()
        stats = sampler.stop()
    return rc, stats


# ---- VAE probe --------------------------------------------------------------


def build_probe_config(
    base_cfg: dict, output_dir: Path, sims: int, repeat: int, data_root: Path,
    batch_size: int | None, world: int = 1,
) -> dict:
    """Shrink a production VAE config to a few optimizer steps, changing nothing
    that affects memory.

    Only the *number* of samples is reduced. Resolution, batch_size, gradient
    checkpointing, mixed precision, the loss set and the unfreeze scope -- every
    knob that sets the activation and optimizer-state footprint -- is left at its
    production value, which is what makes the measured peak transferable.
    """
    cfg = json.loads(json.dumps(base_cfg))  # deep copy through plain types
    if batch_size is not None:
        # The launchers override the config's batch_size (sng_pvc: 4, lrz_ai 2-GPU:
        # 16), and per-rank device memory scales with it -- so the probe has to use
        # the launcher's value, not the config's, or the measurement is for a job
        # nobody runs.
        cfg["optimization"]["batch_size"] = batch_size
    cfg["data"]["data_root"] = str(data_root)
    bs = cfg["optimization"]["batch_size"]
    cfg["data"]["overfit_sims"] = sims
    # The train loader is sharded across ranks with drop_last=True, so the
    # subset must hold at least batch_size * world samples or every rank forms
    # zero batches -- the run then "succeeds" having trained nothing, and only
    # falls over later in loss-curve plotting (KeyError: 'train_rmse') because
    # no per-loss metrics were ever recorded. Two full steps' worth, so the
    # measured peak covers a real optimizer step and not just the first
    # forward.
    need = bs * world * 2
    cfg["data"]["overfit_repeat"] = max(repeat, -(-need // max(1, sims)))
    # VaeTrainer.train() validates `len(dataset) - eval_sims >= 1` *before* it
    # branches into overfit mode, so a production `eval_sims: 675` against a
    # small synthetic probe set fails the guard with "0 train, N eval" long
    # before any memory is allocated. Overfit mode overrides both loaders
    # anyway, so this value is inert past the guard.
    cfg["data"]["eval_sims"] = 1
    cfg["optimization"]["epochs"] = 1
    cfg["optimization"]["gradient_accumulation_steps"] = max(1, repeat // 2)
    cfg["output_dir"] = str(output_dir)
    cfg["clean_output_dir"] = True
    cfg["wandb"]["enabled"] = False
    cfg["visualization"]["num_samples"] = min(cfg["visualization"]["num_samples"], sims)
    return cfg


def probe_vae(
    base_config: Path,
    data_root: Path,
    processes: int,
    sims: int,
    repeat: int,
    workdir: Path,
    batch_size: int | None = None,
    tag: str = "vae",
) -> dict:
    """Run the real VaeTrainer for one short epoch and measure it."""
    base_cfg = yaml.safe_load(base_config.read_text())
    out_dir = workdir / f"{tag}_run"
    run_cfg_path = workdir / f"{tag}_probe_config.yaml"
    cfg = build_probe_config(base_cfg, out_dir, sims, repeat, data_root, batch_size, world=processes)
    run_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    env = os.environ.copy()
    env.setdefault("WINDINET_HF_CACHE", str(REPO_ROOT / "pretrained" / "hub"))
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_MODE"] = "offline"

    mem_prefix = workdir / f"{tag}_mem"
    env["PROBE_MEM_OUT"] = str(mem_prefix)
    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(processes),
        "--num_machines",
        "1",
        "--mixed_precision",
        cfg["acceleration"]["mixed_precision_mode"],
        str(write_mem_shim(workdir)),
        str(REPO_ROOT / "scripts" / "finetune_vae.py"),
        str(run_cfg_path),
    ]

    log = workdir / f"{tag}_probe.log"
    console.print(
        f"[cyan]VAE probe:[/cyan] {processes} rank(s), batch_size {cfg['optimization']['batch_size']}, "
        f"{cfg['data']['overfit_sims']} sims x{repeat}, log -> {log}"
    )
    t0 = time.time()
    rc, stats = run_sampled(cmd, REPO_ROOT, env, log)
    stats["seconds"] = round(time.time() - t0, 1)
    stats["returncode"] = rc
    stats["ranks"] = processes
    stats["log"] = str(log)
    stats["config"] = str(run_cfg_path)
    stats["attempted_batch_size"] = cfg["optimization"]["batch_size"]
    stats.update(collect_rank_memory(mem_prefix))
    stats["batch_size"] = cfg["optimization"]["batch_size"]
    stats["resolution_px"] = None
    stats["resolution"] = f"{cfg.get('_resolution', '128x128')}"

    if rc != 0:
        stats["error"] = log.read_text()[-4000:]
        return stats

    ckpts = sorted((out_dir / "checkpoints").glob("*.safetensors"))
    states = sorted((out_dir / "checkpoints").glob("*.state.pt"))
    stats["checkpoint_bytes"] = ckpts[0].stat().st_size if ckpts else None
    stats["state_bytes"] = states[0].stat().st_size if states else None
    stats["checkpoint_path"] = str(ckpts[0]) if ckpts else None
    stats["best_checkpoint"] = str(out_dir / "checkpoints" / "vae_shockwave_best.safetensors")
    pngs = sorted(out_dir.rglob("frame_*.png"))
    stats["panel_bytes"] = pngs[len(pngs) // 2].stat().st_size if pngs else None
    stats["output_dir"] = str(out_dir)

    # A VAE checkpoint pair is ~15 GB and a batch-size sweep writes one per arm.
    # Their sizes are already recorded, and only the `best` weights are needed
    # downstream by the latent probe -- drop the rest so the probe's own
    # footprint stays a few GB rather than tens.
    for state in (out_dir / "checkpoints").glob("*.state.pt"):
        state.unlink()
    for ckpt in (out_dir / "checkpoints").glob("*.safetensors"):
        if ckpt.name != "vae_shockwave_best.safetensors":
            ckpt.unlink()
    return stats


# ---- preprocessing (latent) probe -------------------------------------------


def probe_preprocess(checkpoint: Path, data_root: Path, workdir: Path, n_sims: int) -> dict:
    """Encode a handful of sims with the probe's own VAE checkpoint.

    Gives the exact on-disk size of one `latents/<id>.pt` + `scalars/<id>.pt`
    pair, which sets both the DiT run's input volume and its file count.
    """
    out_dir = workdir / "preprocessed"
    env = os.environ.copy()
    env.setdefault("WINDINET_HF_CACHE", str(REPO_ROOT / "pretrained" / "hub"))
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "preprocess_dataset.py"),
        str(data_root),
        "--output-dir",
        str(out_dir),
        "--inflate-checkpoint",
        str(checkpoint),
        "--max-samples",
        str(n_sims),
        # --max-samples and --eval-sims are mutually exclusive (truncating the id
        # list would shift the split permutation); the probe only needs latents.
        "--eval-sims",
        "0",
    ]
    log = workdir / "preprocess_probe.log"
    console.print(f"[cyan]Latent probe:[/cyan] encoding {n_sims} sims, log -> {log}")
    rc, stats = run_sampled(cmd, REPO_ROOT, env, log)
    stats["returncode"] = rc
    stats["log"] = str(log)
    if rc != 0:
        stats["error"] = log.read_text()[-4000:]
        return stats

    latents = sorted(out_dir.rglob("latents/*.pt"))
    scalars = sorted(out_dir.rglob("scalars/*.pt"))
    stats["latent_bytes"] = latents[0].stat().st_size if latents else None
    stats["scalar_bytes"] = scalars[0].stat().st_size if scalars else None
    stats["output_dir"] = str(out_dir)
    if latents:
        import torch

        blob = torch.load(latents[0], map_location="cpu", weights_only=False)
        stats["latent_shape"] = list(blob["latents"].shape)
        stats["latent_dtype"] = str(blob["latents"].dtype)
        stats["latent_num_frames"] = int(blob["num_frames"])
        stats["latent_height"] = int(blob["height"])
        stats["latent_width"] = int(blob["width"])
    return stats


# ---- DiT probe --------------------------------------------------------------


def probe_dit(preprocessed: Path, processes: int, steps: int, workdir: Path) -> dict:
    """Run the real DiT trainer for a few optimizer steps on real latents."""
    base_cfg = yaml.safe_load((REPO_ROOT / "configs/dit/train_dit.yaml").read_text())
    corpus = workdir / "dit_data"
    val = corpus / "val"
    for sub in ("latents", "scalars"):
        (corpus / sub).mkdir(parents=True, exist_ok=True)
        (val / sub).mkdir(parents=True, exist_ok=True)

    # Replicate the probe's real latents up to one full effective batch so a
    # step is shaped exactly like production; the values are irrelevant to
    # memory, the shapes are not.
    src_lat = sorted(preprocessed.rglob("latents/*.pt"))
    src_sca = sorted(preprocessed.rglob("scalars/*.pt"))
    if not src_lat:
        return {"returncode": 1, "error": f"no latents under {preprocessed}"}
    need = base_cfg["optimization"]["batch_size"] * base_cfg["optimization"]["gradient_accumulation_steps"] * processes * 2
    for i in range(need):
        s = i % len(src_lat)
        shutil.copy(src_lat[s], corpus / "latents" / f"probe_{i:04d}.pt")
        shutil.copy(src_sca[s], corpus / "scalars" / f"probe_{i:04d}.pt")
    for i in range(2):
        shutil.copy(src_lat[0], val / "latents" / f"probe_val_{i}.pt")
        shutil.copy(src_sca[0], val / "scalars" / f"probe_val_{i}.pt")

    cfg = json.loads(json.dumps(base_cfg))
    cfg["optimization"]["steps"] = steps
    cfg["data"]["preprocessed_data_root"] = str(corpus)
    cfg["validation"]["data_root"] = str(val)
    cfg["validation"]["interval"] = steps  # exercise the eval path exactly once
    cfg["checkpoints"]["interval"] = steps  # and the checkpoint path exactly once
    cfg["checkpoints"]["keep_last_n"] = 2
    cfg["output_dir"] = str(workdir / "dit_run")
    cfg["wandb"]["enabled"] = False
    run_cfg_path = workdir / "dit_probe_config.yaml"
    run_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    env = os.environ.copy()
    env.setdefault("WINDINET_HF_CACHE", str(REPO_ROOT / "pretrained" / "hub"))
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["WANDB_MODE"] = "offline"

    mem_prefix = workdir / "dit_mem"
    env["PROBE_MEM_OUT"] = str(mem_prefix)
    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(processes),
        "--num_machines",
        "1",
        "--mixed_precision",
        cfg["acceleration"]["mixed_precision_mode"],
        str(write_mem_shim(workdir)),
        str(REPO_ROOT / "scripts" / "train.py"),
        str(run_cfg_path),
    ]
    log = workdir / "dit_probe.log"
    console.print(f"[cyan]DiT probe:[/cyan] {processes} rank(s), {steps} steps, log -> {log}")
    t0 = time.time()
    rc, stats = run_sampled(cmd, REPO_ROOT, env, log)
    stats["seconds"] = round(time.time() - t0, 1)
    stats["returncode"] = rc
    stats["ranks"] = processes
    stats["log"] = str(log)
    stats.update(collect_rank_memory(mem_prefix))
    if rc != 0:
        stats["error"] = log.read_text()[-4000:]
        return stats

    out = Path(cfg["output_dir"])
    weights = sorted(out.rglob("model_weights_step_*.safetensors"))
    scalars = sorted(out.rglob("scalar_embedding_step_*.safetensors"))
    stats["checkpoint_bytes"] = weights[0].stat().st_size if weights else None
    stats["scalar_checkpoint_bytes"] = scalars[0].stat().st_size if scalars else None
    stats["output_dir"] = str(out)
    return stats


# ---- activation-scaling probe -----------------------------------------------

# Why this exists: the end-to-end VAE probe's peak barely moves between 128 and
# 256 resolution, which looks wrong until you separate the two contributions.
# The trainer's high-water mark is set by the optimizer step -- fp32 master
# weights + gradients + two AdamW moments for the ~900M trainable parameters of
# the whole-structure unfreeze -- not by the forward/backward activations. This
# probe isolates the activation term alone, so the report can say which one
# dominates at each resolution instead of presenting a flat number that invites
# the wrong conclusion.
_ACT_SHIM = """
import json, os, sys, torch

sys.path.insert(0, os.environ["REPO_ROOT"])
from windinet.inference.model_loader import load_vae
from windinet.vae_adapter import inflate_vae_io_channels

dev = torch.device("cuda:0")
vae = load_vae(os.environ["MODEL_SOURCE"], dtype=torch.bfloat16)
inflate_vae_io_channels(vae, n=4, init="zeros")
vae = vae.to(dev)
vae.enable_gradient_checkpointing()
for p in vae.parameters():
    p.requires_grad_(False)
for p in vae.decoder.parameters():
    p.requires_grad_(True)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
base = torch.cuda.memory_allocated()
rows = []
frames = int(os.environ["FRAMES"])
for res in [int(r) for r in os.environ["RESOLUTIONS"].split(",")]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        x = torch.randn(1, 4, frames, res, res, device=dev, dtype=torch.bfloat16)
        lat = vae.encode(x).latent_dist.mean
        temb = torch.zeros(1, device=dev, dtype=torch.bfloat16)
        rec = vae.decode(lat, temb=temb).sample
        rec.float().pow(2).mean().backward()
        peak = torch.cuda.max_memory_allocated()
        rows.append({"resolution": res, "latent_shape": list(lat.shape),
                     "peak_bytes": peak, "activation_bytes": peak - base})
        del x, lat, rec
    except torch.cuda.OutOfMemoryError:
        rows.append({"resolution": res, "oom": True})
    for p in vae.parameters():
        p.grad = None

json.dump({"weights_bytes": base, "rows": rows}, open(os.environ["ACT_OUT"], "w"))
"""


def vae_config_model_source(config_path: Path) -> str:
    return yaml.safe_load(config_path.read_text())["model"]["model_source"]


def probe_activation_scaling(resolutions: list[int], frames: int, workdir: Path, model_source: str) -> dict:
    """Measure forward+backward activation memory alone, per resolution.

    Runs a single rank with only the decoder trainable and no optimizer, so the
    peak is weights + activations and the activation term can be read off by
    subtraction. `frames` must satisfy T = 8k+1 -- the LTX VAE's temporal
    stride -- which is why the trainer pads its 101-frame clips to 105.
    """
    shim = workdir / "_probe_act_shim.py"
    shim.write_text(_ACT_SHIM)
    out = workdir / "activation_scaling.json"
    env = os.environ.copy()
    env.setdefault("WINDINET_HF_CACHE", str(REPO_ROOT / "pretrained" / "hub"))
    env.update({
        "REPO_ROOT": str(REPO_ROOT),
        "MODEL_SOURCE": model_source,
        "RESOLUTIONS": ",".join(str(r) for r in resolutions),
        "FRAMES": str(frames),
        "ACT_OUT": str(out),
    })
    log = workdir / "activation_probe.log"
    console.print(f"[cyan]Activation probe:[/cyan] {resolutions} at {frames} frames, log -> {log}")
    rc, _ = run_sampled([sys.executable, str(shim)], REPO_ROOT, env, log)
    if rc != 0 or not out.exists():
        return {"returncode": rc, "error": log.read_text()[-3000:]}
    data = json.loads(out.read_text())
    data["returncode"] = 0
    return data


# ----------------------------------------------------------------------------
# 4b. In-process allocator instrumentation
# ----------------------------------------------------------------------------

# nvidia-smi reports the caching allocator's *reserved* pool, which is sticky:
# once PyTorch has grown the pool for the model and optimizer, a larger
# activation footprint can be served from the cache without the reserved figure
# moving. That makes it the right number for provisioning ("this much must be
# free on the card") but a blunt instrument for comparing configurations -- at
# 128 vs 256 resolution it barely moves even though the activations are 4x.
# torch.cuda.max_memory_allocated() is the true live-tensor high-water mark, so
# each rank reports it from an atexit hook and the two are shown side by side.
_MEM_SHIM = '''
import atexit, json, os, runpy, sys
from pathlib import Path

_out = os.environ["PROBE_MEM_OUT"]
_target = sys.argv[1]
sys.argv = sys.argv[1:]


def _dump() -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        Path(f"{_out}.{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "max_allocated": torch.cuda.max_memory_allocated(),
                    "max_reserved": torch.cuda.max_memory_reserved(),
                }
            )
        )
    except Exception:
        pass


atexit.register(_dump)
runpy.run_path(_target, run_name="__main__")
'''


def write_mem_shim(workdir: Path) -> Path:
    shim = workdir / "_probe_mem_shim.py"
    shim.write_text(_MEM_SHIM)
    return shim


def collect_rank_memory(prefix: Path) -> dict:
    """Merge the per-rank allocator reports the shim wrote."""
    reports = []
    for f in sorted(prefix.parent.glob(prefix.name + ".*.json")):
        try:
            reports.append(json.loads(f.read_text()))
        except (OSError, ValueError):
            continue
        f.unlink()
    if not reports:
        return {}
    return {
        "torch_max_allocated_per_rank": max(r["max_allocated"] for r in reports),
        "torch_max_reserved_per_rank": max(r["max_reserved"] for r in reports),
        "torch_ranks_reporting": len(reports),
    }


# ----------------------------------------------------------------------------
# 5. Report
# ----------------------------------------------------------------------------


def _fmt_bytes(n: int | None) -> str:
    if not n:
        return "-"
    for unit, div in (("TB", 1024**4), ("GB", GB), ("MB", MB), ("kB", 1024)):
        if n >= div:
            return f"{n / div:,.2f} {unit}"
    return f"{n} B"


def render(profile: dict) -> None:
    ds = profile["dataset"]
    shapes = profile["job_shapes"]
    probes = profile.get("probes", {})

    console.rule("[bold]Dataset inventory")
    t = Table(show_header=False, box=None)
    t.add_row("Index file", f"{ds['path']}  ({_fmt_bytes(ds['index_file_bytes'])}, HDF5 external-link index)")
    t.add_row("Simulations", f"{ds['n_sims']:,} across {len(ds['gammas'])} gamma values")
    t.add_row("Per simulation", f"{len(ds['fields'])} fields x {ds['frames']} frames x {ds['height']}x{ds['width']} {ds['dtype']}")
    t.add_row("Fields", ", ".join(ds["fields"]))
    t.add_row("Backing shards", f"{len(ds['backing_files'])} files, {_fmt_bytes(sum(ds['backing_files'].values()))} total (gzip)")
    t.add_row("On disk (stored)", _fmt_bytes(ds["stored_bytes"]))
    t.add_row("Uncompressed", f"{_fmt_bytes(ds['logical_bytes'])}  (gzip ratio {ds['compression_ratio']:.2f}x)")
    t.add_row("Per sim", f"{_fmt_bytes(ds['stored_bytes_per_sim'])} stored / {_fmt_bytes(ds['logical_bytes_per_sim'])} in memory")
    console.print(t)

    console.rule("[bold]Q1 -- Processes per job")
    t = Table()
    for col in ("Job", "Cluster", "Nodes", "Ranks/node", "MPI ranks", "IO workers", "OS processes", "CPUs/task", "Eff. batch", "Wall"):
        t.add_column(col)
    for s in shapes:
        t.add_row(
            s["name"], s["cluster"], str(s["nodes"]), str(s["ranks_per_node"]), str(s["ranks"]),
            str(s["dataloader_workers"] * s["ranks"]), str(s["os_processes"]),
            str(s["cpus_per_task"] or "-"), str(s["effective_batch"]), s["wall_limit"],
        )
    console.print(t)
    console.print(
        f"[dim]Average over the {len(shapes)} production job types: "
        f"{profile['q1']['mean_ranks']:.1f} MPI ranks, {profile['q1']['mean_os_processes']:.1f} OS processes "
        f"(1 rank per GPU/tile + {profile['q1']['workers']} dataloader workers each).[/dim]"
    )

    console.rule("[bold]Q2/Q3 -- Memory")
    if not probes:
        console.print("[yellow]No probe run -- rerun with --probe vae (or --probe all) for measured numbers.[/yellow]")
    t = Table()
    for col in ("Probe", "Ranks", "batch", "Host PSS (all)", "Host peak (1 proc)",
                "Host private (1 proc)", "Device live (1 rank)", "Device resv. (1 rank)"):
        t.add_column(col)
    for name, p in probes.items():
        if name.startswith("_"):
            continue
        if p.get("returncode") != 0:
            t.add_row(name, str(p.get("ranks", "-")), "-", "[red]FAILED[/red]", "-", "-", "-", "-")
            continue
        live = p.get("torch_max_allocated_per_rank")
        t.add_row(
            name, str(p.get("ranks", "-")), str(p.get("batch_size", "-")),
            _fmt_bytes(p.get("peak_tree_pss_bytes") or p["peak_tree_rss_bytes"]),
            _fmt_bytes(p["peak_process_rss_bytes"]),
            _fmt_bytes(p.get("peak_process_uss_bytes") or 0),
            _fmt_bytes(live) if live else "[dim]-[/dim]",
            _fmt_bytes(p["peak_process_gpu_bytes"]),
            )
    console.print(t)
    for line in profile["q2q3"]["notes"]:
        console.print(f"  {line}")
    console.print(
        "  [dim]\"Device live\" is torch.cuda.max_memory_allocated() inside the rank -- the true "
        "high-water mark of live tensors, and the figure that responds to resolution and batch size. "
        "\"Device resv.\" is the caching allocator's pool as the driver reports it: the amount that must "
        "be free on the card (the provisioning number), but sticky, so it under-reports differences "
        "between configurations. Host figures are RSS sampled across the process tree at 4 Hz.[/dim]"
    )

    console.rule("[bold]Q4 -- Data transferred to/from the site")
    t = Table()
    t.add_column("Direction"); t.add_column("Item"); t.add_column("Size"); t.add_column("When")
    for row in profile["q4"]["rows"]:
        t.add_row(row["direction"], row["item"], _fmt_bytes(row["bytes"]), row["when"])
    console.print(t)
    console.print(f"  [bold]Total inbound[/bold]  {_fmt_bytes(profile['q4']['inbound_bytes'])}")
    console.print(f"  [bold]Total outbound[/bold] {_fmt_bytes(profile['q4']['outbound_bytes'])}")
    console.print("\n[bold]Resident on project storage[/bold] (never transferred, but must be provisioned):")
    t = Table()
    t.add_column("Item"); t.add_column("Size")
    for row in profile["q4"]["resident_rows"]:
        t.add_row(row["item"], _fmt_bytes(row["bytes"]))
    t.add_row("[bold]Total[/bold]", f"[bold]{_fmt_bytes(profile['q4']['resident_bytes'])}[/bold]")
    console.print(t)

    console.rule("[bold]Q5 -- I/O frequency and size")
    t = Table()
    t.add_column("Stream"); t.add_column("Frequency"); t.add_column("Size per event"); t.add_column("Aggregate")
    for row in profile["q5"]["rows"]:
        t.add_row(row["stream"], row["frequency"], row["size"], row["aggregate"])
    console.print(t)

    console.rule("[bold]Q6 -- Files in a typical production run")
    for stage, items in profile["q6"]["stages"].items():
        console.print(f"\n[bold]{stage}[/bold]")
        t = Table()
        for col in ("File", "Count", "Size each", "Total", "Cadence"):
            t.add_column(col)
        for it in items:
            t.add_row(it["kind"], f"{it['per_run_count']:,}", _fmt_bytes(it["bytes_each"]),
                      _fmt_bytes(it["total_bytes"]), it["cadence"])
        console.print(t)
    console.print(
        f"\n  [bold]One full pipeline pass[/bold] (1 VAE run + preprocessing + 1 DiT run): "
        f"{profile['q6']['total_files']:,} files, {_fmt_bytes(profile['q6']['total_bytes'])} resident."
    )
    console.print(
        f"  [bold]Whole campaign[/bold] ({profile['q6']['n_runs']} VAE runs + 1 DiT run): "
        f"{profile['q6']['campaign_files']:,} files, {_fmt_bytes(profile['q6']['campaign_bytes'])} resident."
    )
    console.print(
        f"  [dim]Cumulative write traffic over one pipeline pass, including checkpoints later "
        f"overwritten or pruned: {_fmt_bytes(profile['q6']['write_traffic_bytes'])}.[/dim]"
    )

    rows = profile.get("resolution_scaling") or []
    if rows:
        console.rule("[bold]Resolution scaling (128 -> 256 -> 512)")
        t = Table()
        for col in ("Resolution", "Latent grid", "Tokens/sim", "Latent .pt", "Train set (real)",
                    "Per sim", "Read/epoch", "Device live/rank", "Device resv./rank", "at batch"):
            t.add_column(col)
        for r in rows:
            dev = _fmt_bytes(r["device_live_per_rank"]) if r.get("device_live_per_rank") else (
                "[red]failed[/red]" if r["oom_arms"] else "[dim]not probed[/dim]")
            t.add_row(
                r["resolution"], r["latent_grid"], f"{r['tokens']:,}", _fmt_bytes(r["latent_bytes"]),
                _fmt_bytes(r["train_bytes"]), _fmt_bytes(r["per_sim_bytes"]),
                _fmt_bytes(r["per_epoch_bytes"]), dev,
                _fmt_bytes(r["device_per_rank"]) if r["device_per_rank"] else "-",
                (f"{r['measured_batch']}" if r["measured_batch"] else "-")
                + (f" [red](OOM at {r['oom_batches'][0]})[/red]" if r.get("oom_batches") else ""),
            )
        console.print(t)
        console.print(
            "  [dim]Train-set sizes are the real published shard sizes"
            + (" (fetched live from the HuggingFace dataset repo)." if profile.get("hf_sizes_live")
               else " (recorded from the dataset repo; no outbound access this run).")
            + " Latent geometry is exact (the VAE's fixed 32x spatial / 8x temporal compression),"
            " not extrapolated. Device/host per rank are measured against a synthetic dataset of"
            " identical shape at that resolution.[/dim]"
        )

    act = profile.get("activation_scaling") or {}
    if act.get("returncode") == 0:
        console.print(
            f"\n[bold]Where the VAE's device memory actually goes[/bold] "
            f"(single rank, batch 1, bf16, gradient checkpointing on; "
            f"{_fmt_bytes(act['weights_bytes'])} of resident weights excluded):"
        )
        t = Table()
        for col in ("Resolution", "Latent", "Forward+backward peak", "Activations alone", "vs 128"):
            t.add_column(col)
        base_act = next((r["activation_bytes"] for r in act["rows"] if r.get("resolution") == 128), None)
        for r in act["rows"]:
            if r.get("oom"):
                t.add_row(f"{r['resolution']}x{r['resolution']}", "-", "[red]OOM[/red]", "-", "-")
                continue
            ratio = f"{r['activation_bytes'] / base_act:.1f}x" if base_act else "-"
            t.add_row(
                f"{r['resolution']}x{r['resolution']}",
                "x".join(str(d) for d in r["latent_shape"]),
                _fmt_bytes(r["peak_bytes"]), _fmt_bytes(r["activation_bytes"]), ratio,
            )
        console.print(t)
        console.print(
            "  [dim]This is why the end-to-end VAE figure barely moves from 128 to 256: the trainer's "
            "high-water mark is the optimizer step (fp32 master weights + gradients + two AdamW moments "
            "for the whole-structure unfreeze), not the activations. Activations only become the "
            "dominant term at 512.[/dim]"
        )

    console.rule("[bold green]Answers, for the proposal form")
    for n, line in profile["answers"]:
        console.print(f"[bold]{n}.[/bold] {line}\n")


# ----------------------------------------------------------------------------
# 6. Assembly
# ----------------------------------------------------------------------------


def assemble(
    ds: DatasetProfile,
    shapes: list[JobShape],
    probes: dict,
    vae_config: Path,
    n_runs: int,
) -> dict:
    vae_cfg = yaml.safe_load(vae_config.read_text())
    dit_cfg = yaml.safe_load((REPO_ROOT / "configs/dit/train_dit.yaml").read_text())

    # The largest successful VAE batch size measured is the production-representative one.
    vae_candidates = [v for k, v in probes.items() if k.startswith("VAE finetune") and v.get("returncode") == 0]
    res_probes = {k: v for k, v in probes.items() if k.startswith("VAE ") and "x" in k.split()[1]}
    vae_probe = max(vae_candidates, key=lambda v: v.get("batch_size", 0)) if vae_candidates else {}
    pre_probe = probes.get("Latent preprocessing", {})
    dit_probe = probes.get("DiT training", {})

    ckpt_bytes = vae_probe.get("checkpoint_bytes")
    state_bytes = vae_probe.get("state_bytes")
    png_bytes = vae_probe.get("panel_bytes") or measured_png_bytes()
    latent_bytes = pre_probe.get("latent_bytes")
    scalar_bytes = pre_probe.get("scalar_bytes")
    dit_ckpt_bytes = dit_probe.get("checkpoint_bytes")
    dit_scalar_bytes = dit_probe.get("scalar_checkpoint_bytes")

    # --- Q1
    mean_ranks = sum(s.ranks for s in shapes) / len(shapes)
    mean_procs = sum(s.os_processes for s in shapes) / len(shapes)

    # --- Q2/Q3
    notes = []
    vae_shape = next((s for s in shapes if "VAE" in s.name and s.cluster == "sng_pvc"), shapes[0])
    if vae_probe.get("returncode") == 0:
        per_rank_rss = vae_probe["peak_process_rss_bytes"]
        per_rank_gpu = vae_probe["peak_process_gpu_bytes"]
        measured_ranks = vae_probe["ranks"]
        # Host RSS does not scale purely per-rank (page cache and the HDF5 read
        # path are shared), so scale the *measured tree* by rank count rather
        # than multiplying a single rank's peak.
        scaled_host = vae_probe["peak_tree_rss_bytes"] / measured_ranks * vae_shape.ranks
        scaled_dev = per_rank_gpu * vae_shape.ranks
        notes.append(
            f"[bold]VAE finetune at production width[/bold] ({vae_shape.ranks} ranks, {vae_shape.launcher}): "
            f"host {_fmt_bytes(int(scaled_host))}, device {_fmt_bytes(int(scaled_dev))} "
            f"-- scaled from the {measured_ranks}-rank measurement above."
        )
        notes.append(
            f"Max per process: [bold]{per_rank_rss / MB:,.0f} MB[/bold] host RSS, "
            f"{per_rank_gpu / MB:,.0f} MB device, per training rank. Dataloader workers are much "
            "smaller (they hold one decompressed sim each)."
        )
        job_host = int(scaled_host)
        job_dev = int(scaled_dev)
        max_proc = per_rank_rss
    else:
        notes.append("[yellow]VAE memory not measured; rerun with --probe vae.[/yellow]")
        job_host = job_dev = max_proc = 0

    if dit_probe.get("returncode") == 0:
        dit_shape = next((s for s in shapes if "DiT" in s.name), None)
        n = dit_shape.ranks if dit_shape else dit_probe["ranks"]
        scaled = dit_probe["peak_tree_rss_bytes"] / dit_probe["ranks"] * n
        notes.append(
            f"[bold]DiT training at production width[/bold] ({n} ranks): host {_fmt_bytes(int(scaled))}, "
            f"device {_fmt_bytes(dit_probe['peak_process_gpu_bytes'] * n)} "
            f"({dit_probe['peak_process_gpu_bytes'] / MB:,.0f} MB per rank -- fp32 weights + grads + "
            "AdamW moments for the 1.9B transformer dominate)."
        )
        max_proc = max(max_proc, dit_probe["peak_process_rss_bytes"])
        job_host = max(job_host, int(scaled))
        job_dev = max(job_dev, dit_probe["peak_process_gpu_bytes"] * n)

    # --- Q4
    weights_dir = REPO_ROOT / "pretrained" / "hub"
    weights_bytes = (
        sum(f.stat().st_size for f in weights_dir.rglob("*") if f.is_file() and not f.is_symlink())
        if weights_dir.exists()
        else LTXV_WEIGHTS_FALLBACK_BYTES
    )

    n_latent_pairs = ds.n_sims
    latent_total = (latent_bytes or 0) * n_latent_pairs + (scalar_bytes or 0) * n_latent_pairs

    vae_items = vae_output_inventory(vae_cfg, ckpt_bytes, state_bytes, png_bytes)
    dit_items = dit_output_inventory(dit_cfg, dit_ckpt_bytes, dit_scalar_bytes)
    vae_run_bytes = sum(i.total_bytes for i in vae_items)
    dit_run_bytes = sum(i.total_bytes for i in dit_items)

    # What actually crosses the site boundary is not the same as what the run
    # writes: every launcher mirrors its output back with
    # `rsync -a --exclude='checkpoints/'`, so weights stay on cluster scratch and
    # only diagnostics (plus the one checkpoint promoted to the next stage) move.
    ckpt_pair = (ckpt_bytes or 0) + (state_bytes or 0)
    diagnostics_per_run = sum(
        i.total_bytes for i in vae_items if not i.kind.startswith("checkpoints/")
    )

    q4_rows = [
        {"direction": "IN", "item": f"Euler-MQ dataset ({ds.n_sims:,} sims, {len(ds.backing_files)} HDF5 shards)",
         "bytes": ds.stored_bytes, "when": "once, staged to project storage"},
        {"direction": "IN", "item": "LTX-Video 2B pretrained weights (HF hub cache)",
         "bytes": weights_bytes, "when": "once per machine"},
        {"direction": "IN", "item": "Source checkout (git)", "bytes": 30 * MB, "when": "once, then incremental pulls"},
        {"direction": "OUT", "item": f"Diagnostics mirror -- panels, metrics, configs, logs ({n_runs} runs)",
         "bytes": diagnostics_per_run * n_runs,
         "when": "rsync at the end of each run (checkpoints/ excluded by the launchers)"},
        {"direction": "OUT", "item": "Promoted VAE checkpoints archived off-cluster",
         "bytes": (ckpt_bytes or 0) * 3, "when": "only for runs that become a reference baseline"},
        {"direction": "OUT", "item": "Final DiT checkpoint + sample rollouts",
         "bytes": (dit_ckpt_bytes or 0) + 500 * MB, "when": "once at the end of the DiT run"},
    ]

    resident_rows = [
        {"item": f"Dataset, resolutions in use ({ds.n_sims:,} sims at {ds.height}x{ds.width}; "
                 "a 256x256 copy is ~4x)", "bytes": ds.stored_bytes},
        {"item": f"VAE checkpoints, {n_runs} concurrent/archived runs x (weights + optimizer state)",
         "bytes": ckpt_pair * n_runs},
        {"item": f"Preprocessed latents ({ds.n_sims:,} pairs)", "bytes": latent_total},
        {"item": "DiT checkpoint rolling window",
         "bytes": (dit_ckpt_bytes or 0) * dit_cfg["checkpoints"]["keep_last_n"]},
        {"item": f"Diagnostics ({n_runs} runs)", "bytes": diagnostics_per_run * n_runs},
        {"item": "Pretrained weight cache", "bytes": weights_bytes},
    ]

    inbound = sum(r["bytes"] for r in q4_rows if r["direction"] == "IN")
    outbound = sum(r["bytes"] for r in q4_rows if r["direction"] == "OUT")

    # --- Q5
    epochs = vae_cfg["optimization"]["epochs"]
    eval_sims = vae_cfg["data"]["eval_sims"]
    train_sims = ds.n_sims - eval_sims
    # File bytes, not the sum of per-dataset compressed payloads: HDF5 chunk
    # index/metadata overhead means the shards are ~82 GB on disk while their
    # datasets sum to ~75 GB. What the filesystem actually serves each epoch is
    # the former, and it is also the basis the resolution table uses, so both
    # report the same number.
    per_epoch_read = ds.stored_bytes
    dit_steps = dit_cfg["optimization"]["steps"]
    dit_ckpt_interval = dit_cfg["checkpoints"]["interval"]
    dit_writes = dit_steps // dit_ckpt_interval
    dit_eff_batch = (
        dit_cfg["optimization"]["batch_size"] * dit_cfg["optimization"]["gradient_accumulation_steps"]
    )

    q5_rows = [
        {"stream": "HDF5 read (VAE train+eval)",
         "frequency": f"every epoch, {epochs} epochs; {train_sims:,} train + {eval_sims} eval sims",
         "size": f"{_fmt_bytes(ds.stored_bytes_per_sim)}/sim compressed, {_fmt_bytes(ds.logical_bytes_per_sim)} decompressed",
         "aggregate": f"{_fmt_bytes(per_epoch_read)}/epoch, {_fmt_bytes(per_epoch_read * epochs)} per run"},
        {"stream": "VAE checkpoint write",
         "frequency": f"every {vae_cfg['checkpoints']['interval']} epoch(s)",
         "size": f"{_fmt_bytes(ckpt_bytes)} weights + {_fmt_bytes(state_bytes)} optimizer state",
         "aggregate": f"{_fmt_bytes(((ckpt_bytes or 0) + (state_bytes or 0)) * epochs)} written over the run "
                      f"into {2 + len(vae_cfg['checkpoints'].get('save_epochs', []))} rewritten slots"},
        {"stream": "Reconstruction panels",
         "frequency": f"every {vae_cfg['visualization']['interval_epochs']} epoch(s), rank 0",
         "size": f"{_fmt_bytes(png_bytes)} x {vae_cfg['visualization']['num_samples'] * len(vae_cfg['visualization']['frame_numbers'])} PNGs",
         "aggregate": _fmt_bytes((png_bytes or 0) * epochs * vae_cfg["visualization"]["num_samples"] * len(vae_cfg["visualization"]["frame_numbers"]))},
        {"stream": "Latent preprocessing (one-off, between stages)",
         "frequency": "once per VAE checkpoint promoted to DiT training",
         "size": f"reads {_fmt_bytes(ds.stored_bytes)}, writes {_fmt_bytes(latent_bytes)} + {_fmt_bytes(scalar_bytes)} per sim",
         "aggregate": f"{_fmt_bytes(latent_total)} of latents for {ds.n_sims:,} sims"},
        {"stream": "Latent read (DiT train)",
         "frequency": f"every step; {dit_eff_batch} samples/step x {dit_steps:,} steps",
         "size": _fmt_bytes(latent_bytes),
         "aggregate": f"{_fmt_bytes((latent_bytes or 0) * dit_eff_batch * dit_steps)} of reads, "
                      f"but the whole {_fmt_bytes(latent_total)} corpus fits in page cache"},
        {"stream": "DiT checkpoint write",
         "frequency": f"every {dit_ckpt_interval} optimizer steps ({dit_writes} writes)",
         "size": _fmt_bytes(dit_ckpt_bytes),
         "aggregate": f"{_fmt_bytes((dit_ckpt_bytes or 0) * dit_writes)} of write traffic; "
                      f"{_fmt_bytes((dit_ckpt_bytes or 0) * dit_cfg['checkpoints']['keep_last_n'])} resident "
                      f"(keep_last_n={dit_cfg['checkpoints']['keep_last_n']})"},
        {"stream": "Slurm stdout/stderr",
         "frequency": "continuous",
         "size": "~1 line/step",
         "aggregate": "< 5 MB per job"},
    ]

    # --- Q6
    stages = {
        f"Stage 1 -- VAE finetuning, one run ({epochs} epochs, {vae_config.name})": [
            {**i.__dict__, "total_bytes": i.total_bytes} for i in vae_items
        ],
        f"Stage 2 -- Latent preprocessing, {ds.n_sims:,} sims": [
            {"kind": "latents/<sim_id>.pt", "per_run_count": ds.n_sims, "bytes_each": latent_bytes or 0,
             "total_bytes": (latent_bytes or 0) * ds.n_sims, "cadence": "one per simulation, written once", "note": ""},
            {"kind": "scalars/<sim_id>.pt", "per_run_count": ds.n_sims, "bytes_each": scalar_bytes or 0,
             "total_bytes": (scalar_bytes or 0) * ds.n_sims, "cadence": "one per simulation, written once", "note": ""},
            {"kind": "normalization.json + split_manifest.json", "per_run_count": 2, "bytes_each": 150_000,
             "total_bytes": 300_000, "cadence": "once", "note": ""},
        ],
        f"Stage 3 -- DiT training, one run ({dit_steps:,} steps)": [
            {**i.__dict__, "total_bytes": i.total_bytes} for i in dit_items
        ],
    }
    total_files = sum(it["per_run_count"] for items in stages.values() for it in items)
    total_bytes = sum(it["total_bytes"] for items in stages.values() for it in items)
    write_traffic = ckpt_pair * epochs + (dit_ckpt_bytes or 0) * dit_writes
    vae_files_per_run = sum(it["per_run_count"] for it in stages[list(stages)[0]])
    vae_bytes_per_run = sum(it["total_bytes"] for it in stages[list(stages)[0]])

    # --- resolution scaling
    # The LTX-Video VAE compresses 32x spatially and 8x temporally into 128
    # latent channels, so the token count per simulation is fixed by geometry:
    # (H/32)*(W/32)*ceil((T-1)/8 + 1). That part is exact, not extrapolated.
    hf_sizes = fetch_hf_dataset_sizes()
    # Anchor the token count on the measured latent rather than on a formula:
    # the VAE pads the 101-frame clip up to its temporal stride, so the naive
    # (T-1)//8 + 1 undercounts (13 vs the 14 latent frames actually produced).
    measured_shape = pre_probe.get("latent_shape")
    native_grid = ds.height // 32
    if measured_shape:
        native_tokens = measured_shape[0]
        latent_frames = native_tokens // (native_grid * native_grid)
    else:
        latent_frames = (ds.frames - 1) // 8 + 1
        native_tokens = native_grid * native_grid * latent_frames
    res_rows = []
    for res in sorted(set(list(RESOLUTION_CONFIGS) + [ds.height])):
        grid = res // 32
        tokens = grid * grid * latent_frames
        scale = (res / ds.height) ** 2
        train_bytes = (
            hf_sizes[res]["train_bytes"] if hf_sizes and res in hf_sizes
            else HF_TRAIN_BYTES.get(res, int(ds.stored_bytes * scale))
        )
        measured = [
            (k, v) for k, v in probes.items()
            if k.startswith(f"VAE {res}x{res}") and v.get("returncode") == 0
        ]
        oomed = [
            k for k, v in probes.items()
            if k.startswith(f"VAE {res}x{res}") and v.get("returncode") not in (0, None)
        ]
        best = max(measured, key=lambda kv: kv[1].get("batch_size", 0)) if measured else None
        res_rows.append({
            "resolution": f"{res}x{res}",
            "latent_grid": f"{grid}x{grid}x{latent_frames}",
            "tokens": tokens,
            "latent_bytes": int((latent_bytes or 0) * tokens / native_tokens),
            "train_bytes": train_bytes,
            "per_sim_bytes": train_bytes // ds.n_sims,
            "uncompressed_bytes": int(ds.logical_bytes * scale),
            "per_epoch_bytes": train_bytes,
            "device_per_rank": best[1]["peak_process_gpu_bytes"] if best else None,
            "device_live_per_rank": best[1].get("torch_max_allocated_per_rank") if best else None,
            "host_per_rank": best[1]["peak_process_rss_bytes"] if best else None,
            "measured_batch": best[1].get("batch_size") if best else None,
            "oom_arms": oomed,
            # Batch sizes that failed at this resolution, so the table can say
            # "batch 1, and batch 4 OOMs" instead of silently reporting the
            # largest arm that happened to survive.
            "oom_batches": sorted(
                probes[k].get("attempted_batch_size") for k in oomed
                if probes[k].get("attempted_batch_size")
            ),
            "latent_corpus_bytes": int((latent_bytes or 0) * tokens / native_tokens * ds.n_sims),
        })

    # --- condensed answers, phrased the way a proposal form wants them
    ranks_min = min(s.ranks for s in shapes)
    ranks_max = max(s.ranks for s in shapes)
    procs_min = min(s.os_processes for s in shapes)
    procs_max = max(s.os_processes for s in shapes)
    per_rank_host = vae_probe.get("peak_process_rss_bytes", 0)
    per_rank_dev = vae_probe.get("peak_process_gpu_bytes", 0)
    dit_rank_dev = dit_probe.get("peak_process_gpu_bytes", 0)
    dit_rank_host = dit_probe.get("peak_process_rss_bytes", 0)
    dit_rank_host_tree = (
        dit_probe.get("peak_tree_rss_bytes", 0) / dit_probe.get("ranks", 1) * vae_shape.ranks
        if dit_probe.get("ranks") else 0
    )

    unmeasured = "[yellow]Not measured -- rerun with --probe all.[/yellow]"

    answers = [
        (
            "Average number of processes",
            f"One MPI/DDP rank per GPU or XPU tile, plus {vae_shape.dataloader_workers} dataloader "
            f"worker processes per rank. Production jobs run {ranks_min}-{ranks_max} ranks on a single "
            f"node ({procs_min}-{procs_max} OS processes); the average across the {len(shapes)} "
            f"job types in use is [bold]{mean_ranks:.1f} ranks / {mean_procs:.1f} processes[/bold], and "
            f"the largest configuration is {ranks_max} ranks / {procs_max} processes. Multi-node is "
            "supported by the launchers but not yet in production use.",
        ),
        (
            "Average job memory (total over all nodes)",
            unmeasured if not vae_probe else
            f"Both stages run on a single node. VAE finetuning at {vae_shape.ranks} ranks: "
            f"[bold]~{vae_probe.get('peak_tree_rss_bytes', 0) / vae_probe.get('ranks', 1) * vae_shape.ranks / GB:.0f} GB "
            f"host RAM[/bold] and "
            f"[bold]~{per_rank_dev * vae_shape.ranks / GB:.0f} GB device memory[/bold], scaled from a "
            f"measured {vae_probe.get('ranks', '?')}-rank run "
            f"({_fmt_bytes(vae_probe.get('peak_tree_rss_bytes', 0))} host, "
            f"{_fmt_bytes(vae_probe.get('peak_tree_gpu_bytes', 0))} device). DiT training at the same "
            f"width: [bold]~{dit_rank_host_tree / GB:.0f} GB host[/bold] and "
            f"[bold]~{dit_rank_dev * vae_shape.ranks / GB:.0f} GB device[/bold]. Smaller 2-GPU jobs -- "
            f"the lrz_ai default -- are what was measured directly: "
            f"{_fmt_bytes(vae_probe.get('peak_tree_rss_bytes', 0))} host and "
            f"{_fmt_bytes(vae_probe.get('peak_tree_gpu_bytes', 0))} device for the VAE, "
            f"{_fmt_bytes(dit_probe.get('peak_tree_rss_bytes', 0))} / "
            f"{_fmt_bytes(dit_probe.get('peak_tree_gpu_bytes', 0))} for the DiT. Device memory scales "
            "essentially linearly with rank count (data parallelism, full model replica per rank); host "
            "RAM scales slightly sublinearly.",
        ),
        (
            "Maximum memory per process",
            unmeasured if not vae_probe else
            f"[bold]{per_rank_host / MB:,.0f} MB[/bold] host RSS per training rank for VAE finetuning "
            f"(batch {vae_probe.get('batch_size', '?')}), rising to "
            f"[bold]{dit_rank_host / MB:,.0f} MB[/bold] for a DiT rank. On the device: "
            f"{per_rank_dev / MB:,.0f} MB per VAE rank and {dit_rank_dev / MB:,.0f} MB per DiT rank. "
            "The DiT figure was reserved on a 48 GB card, so it is an upper bound rather than a proven "
            f"floor -- the irreducible part is ~{31 * GB / GB:.0f} GB of fp32 weights + gradients + "
            "AdamW moments for the 1.9B transformer, so a 40 GB-class GPU is the practical minimum and "
            "48 GB is what has actually been exercised. Dataloader workers add a few hundred MB each.",
        ),
        (
            "Total data transferred to/from",
            f"[bold]{inbound / GB:.0f} GB inbound[/bold] once at project setup "
            f"({_fmt_bytes(ds.stored_bytes)} of Euler-MQ HDF5 + {_fmt_bytes(weights_bytes)} of "
            f"pretrained LTX-Video weights), then only incremental git pulls. "
            f"[bold]{outbound / GB:.0f} GB outbound[/bold] over the campaign: the launchers mirror back "
            "diagnostics only (`rsync --exclude=checkpoints/`), so weights leave the site only for the "
            f"handful of runs promoted to a reference baseline. Resident on project storage: "
            f"[bold]{sum(r['bytes'] for r in resident_rows) / GB:.0f} GB[/bold] "
            f"({n_runs} runs' checkpoints dominate; a 256x256 dataset copy would add ~"
            f"{4 * ds.stored_bytes / GB:.0f} GB).",
        ),
        (
            "Frequency and size of data output/input",
            f"Input: the full {_fmt_bytes(ds.stored_bytes)} dataset is streamed once per epoch "
            f"({_fmt_bytes(ds.stored_bytes_per_sim)} compressed per simulation, "
            f"{_fmt_bytes(ds.logical_bytes_per_sim)} decompressed), i.e. "
            f"~{per_epoch_read / GB:.0f} GB/epoch and {_fmt_bytes(per_epoch_read * epochs)} over a "
            f"{epochs}-epoch run. Output: a "
            f"{_fmt_bytes((ckpt_bytes or 0) + (state_bytes or 0))} checkpoint (weights + optimizer "
            f"state) every {vae_cfg['checkpoints']['interval']} epoch(s) into a fixed set of slots, plus "
            f"{vae_cfg['visualization']['num_samples'] * len(vae_cfg['visualization']['frame_numbers'])} "
            f"PNG panels of ~{_fmt_bytes(png_bytes)} and one metrics-CSV append per epoch. The DiT stage "
            f"writes a {_fmt_bytes(dit_ckpt_bytes)} checkpoint every {dit_ckpt_interval} steps "
            f"({dit_writes} writes, {_fmt_bytes((dit_ckpt_bytes or 0) * dit_writes)} of traffic) while "
            f"reading a {_fmt_bytes(latent_total)} latent corpus that stays in page cache. "
            "I/O is bursty, not continuous: checkpoint writes are the only large events.",
        ),
        (
            "Files and sizes in a typical production run",
            f"One VAE run leaves {vae_files_per_run:,} files / {_fmt_bytes(vae_bytes_per_run)}: "
            f"{2 + len(vae_cfg['checkpoints'].get('save_epochs', []))} checkpoints of "
            f"{_fmt_bytes(ckpt_bytes)} each, one {_fmt_bytes(state_bytes)} optimizer-state file, "
            f"{epochs * vae_cfg['visualization']['num_samples'] * len(vae_cfg['visualization']['frame_numbers'])} "
            f"PNGs of ~{_fmt_bytes(png_bytes)}, and a handful of small CSV/YAML/log files. "
            f"Preprocessing adds {2 * ds.n_sims:,} small `.pt` files ({_fmt_bytes(latent_bytes)} and "
            f"{_fmt_bytes(scalar_bytes)}) totalling {_fmt_bytes(latent_total)} -- the only large *file "
            f"count* in the project. The DiT run keeps a rolling window of "
            f"{dit_cfg['checkpoints']['keep_last_n']} x {_fmt_bytes(dit_ckpt_bytes)}. "
            f"Whole campaign ({n_runs} VAE runs + 1 DiT run): "
            f"~{vae_files_per_run * n_runs + (total_files - vae_files_per_run):,} files, "
            f"{_fmt_bytes(vae_bytes_per_run * n_runs + (total_bytes - vae_bytes_per_run))}.",
        ),
    ]

    if res_rows and any(r.get("device_live_per_rank") for r in res_rows):
        by_res = {r["resolution"]: r for r in res_rows}
        native = f"{ds.height}x{ds.width}"
        parts = []
        for key in ("256x256", "512x512"):
            r = by_res.get(key)
            if not r or key == native:
                continue
            ratio = r["train_bytes"] / by_res[native]["train_bytes"]
            oom = (f", and batch {r['oom_batches'][0]} OOMs on a 48 GB card"
                   if r.get("oom_batches") else "")
            parts.append(
                f"[bold]{key}[/bold]: dataset {_fmt_bytes(r['train_bytes'])} ({ratio:.1f}x), so "
                f"~{r['per_epoch_bytes'] / GB:.0f} GB/epoch and "
                f"{_fmt_bytes(r['per_epoch_bytes'] * epochs)} per {epochs}-epoch run; latents grow to "
                f"{_fmt_bytes(r['latent_bytes'])} each ({_fmt_bytes(r['latent_corpus_bytes'])} corpus); "
                f"per-rank device {_fmt_bytes(r['device_live_per_rank'])} live at batch "
                f"{r['measured_batch']}{oom}."
            )
        if parts:
            answers.append((
                "Same answers at higher resolution",
                "Process counts, file counts and checkpoint sizes are unchanged -- the trainable "
                "parameter count is fixed by the architecture, not the input resolution, so Q1 and "
                "Q6's file inventory carry over verbatim. What scales is data volume (with pixel "
                "count) and device memory (not linearly). " + " ".join(parts) +
                " Host RAM is essentially flat across all three.",
            ))

    return {
        "answers": answers,
        "resolution_scaling": res_rows,
        "activation_scaling": probes.get("_activation_scaling", {}),
        "hf_sizes_live": bool(hf_sizes),
        "dataset": {**ds.__dict__, "path": str(ds.path),
                    "logical_bytes": ds.logical_bytes, "stored_bytes": ds.stored_bytes,
                    "compression_ratio": ds.compression_ratio},
        "job_shapes": [{**s.__dict__, "ranks": s.ranks, "os_processes": s.os_processes,
                        "effective_batch": s.effective_batch} for s in shapes],
        "probes": probes,
        "q1": {"mean_ranks": mean_ranks, "mean_os_processes": mean_procs,
               "workers": vae_shape.dataloader_workers},
        "q2q3": {"job_host_bytes": job_host, "job_device_bytes": job_dev,
                 "max_process_bytes": max_proc, "notes": notes},
        "q4": {"rows": q4_rows, "inbound_bytes": inbound, "outbound_bytes": outbound,
               "resident_rows": resident_rows,
               "resident_bytes": sum(r["bytes"] for r in resident_rows)},
        "q5": {"rows": q5_rows},
        "q6": {"stages": stages, "total_files": total_files, "total_bytes": total_bytes,
               "write_traffic_bytes": write_traffic, "n_runs": n_runs,
               "campaign_files": vae_files_per_run * n_runs + (total_files - vae_files_per_run),
               "campaign_bytes": vae_bytes_per_run * n_runs + (total_bytes - vae_bytes_per_run)},
    }


@app.command()
def main(
    data_root: Path = typer.Option(
        REPO_ROOT / "euler_mq_dataset/128x128_ds/train.h5",
        "--data-root", help="Euler-MQ train.h5 index to inventory.",
    ),
    vae_config: Path = typer.Option(
        REPO_ROOT / "configs/finetune_vae/finetune_vae_whole_structure_baseline_ep30.yaml",
        "--vae-config", help="Production VAE config whose cadence/epoch budget defines a run.",
    ),
    probe: str = typer.Option(
        "none", "--probe", help="none | vae | dit | all. Runs the real trainers briefly and measures them.",
    ),
    probe_processes: int = typer.Option(2, "--probe-processes", help="Ranks to launch for the probe."),
    probe_sims: int = typer.Option(2, "--probe-sims", help="Simulations the VAE probe trains on."),
    probe_batch_sizes: str = typer.Option(
        "1,4", "--probe-batch-sizes",
        help="Comma-separated per-rank batch sizes to measure the VAE at. Production launchers use "
             "4 (sng_pvc) and 16 (lrz_ai 2-GPU); device memory scales with this, so it is measured, "
             "not assumed.",
    ),
    probe_repeat: int = typer.Option(4, "--probe-repeat", help="Times the probe subset repeats per epoch."),
    dit_steps: int = typer.Option(4, "--dit-steps", help="Optimizer steps for the DiT probe."),
    resolutions: str = typer.Option(
        "", "--resolutions",
        help="Comma-separated resolutions to measure the VAE at, e.g. '128,256,512'. Anything other "
             "than the native resolution of --data-root is probed against a synthetic upsampled "
             "dataset of the same shape (see synthesize_resolution). Implies --probe vae.",
    ),
    resolution_sims: int = typer.Option(
        2, "--resolution-sims", help="Simulations to synthesize per non-native resolution.",
    ),
    runs_per_campaign: int = typer.Option(
        20, "--runs", help="VAE runs in a typical campaign, for the Q4 outbound total.",
    ),
    workdir: Path = typer.Option(None, "--workdir", help="Where probe artifacts land (default: a temp dir)."),
    json_out: Path = typer.Option(None, "--json", help="Also write the full profile as JSON."),
) -> None:
    """Profile WinDiNet's compute/storage footprint for a resource proposal."""
    if not data_root.exists():
        console.print(f"[red]No such dataset:[/red] {data_root}")
        raise typer.Exit(1)

    console.print(f"[dim]Scanning {data_root} ...[/dim]")
    ds = scan_dataset(data_root)
    shapes = read_job_shapes()

    probes: dict = {}
    if resolutions and probe == "none":
        probe = "vae"
    if probe != "none":
        workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="windinet_profile_"))
        workdir.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(workdir).free
        console.print(f"[dim]Probe artifacts -> {workdir}  ({_fmt_bytes(free)} free)[/dim]")
        if free < 40 * GB:
            # A VAE weights+state pair is ~15 GB and the DiT checkpoint ~7 GB;
            # a quota-exceeded failure halfway through wastes the whole probe.
            console.print(
                f"[yellow]Warning:[/yellow] {_fmt_bytes(free)} free at {workdir}. The probe writes real "
                "checkpoints (~15 GB for the VAE pair, ~7 GB for the DiT) -- pass --workdir on a volume "
                "with 40 GB+ free."
            )

        vae_stats = None
        for bs in [int(x) for x in probe_batch_sizes.split(",") if x.strip()]:
            stats = probe_vae(
                vae_config, data_root, probe_processes, probe_sims, probe_repeat, workdir,
                batch_size=bs, tag=f"vae_bs{bs}",
            )
            probes[f"VAE finetune (batch {bs})"] = stats
            # The largest batch that ran is the one that represents production.
            if stats.get("returncode") == 0:
                if vae_stats is not None and vae_stats.get("output_dir"):
                    shutil.rmtree(vae_stats["output_dir"], ignore_errors=True)
                vae_stats = stats
        if vae_stats is None:
            vae_stats = {"returncode": 1}

        res_list = [int(x) for x in resolutions.split(",") if x.strip()]
        if res_list:
            # T must be 8k+1 for the LTX VAE's temporal stride; the trainer pads
            # its 101-frame clips to 105, so match that here.
            padded = ((ds.frames - 2) // 8 + 1) * 8 + 1
            probes["_activation_scaling"] = probe_activation_scaling(
                res_list, padded, workdir, vae_config_model_source(vae_config)
            )

        for res in res_list:
            cfg_rel = RESOLUTION_CONFIGS.get(res)
            if cfg_rel is None:
                console.print(f"[yellow]No config registered for {res}x{res}; skipping.[/yellow]")
                continue
            if res == ds.height:
                res_data = data_root
            else:
                res_data = workdir / f"synth_{res}" / "train.h5"
                if not res_data.exists():
                    console.print(f"[dim]Synthesizing {resolution_sims} sims at {res}x{res} ...[/dim]")
                    synthesize_resolution(data_root, res_data, res, resolution_sims)
            for bs in [int(x) for x in probe_batch_sizes.split(",") if x.strip()]:
                probes[f"VAE {res}x{res} (batch {bs})"] = probe_vae(
                    REPO_ROOT / cfg_rel, res_data, probe_processes,
                    min(probe_sims, resolution_sims), 2, workdir,
                    batch_size=bs, tag=f"vae_{res}_bs{bs}",
                )

        if probe in ("dit", "all") and vae_stats.get("returncode") == 0:
            pre = probe_preprocess(Path(vae_stats["best_checkpoint"]), data_root, workdir,
                                   n_sims=max(8, probe_processes * 4))
            probes["Latent preprocessing"] = pre
            if pre.get("returncode") == 0:
                probes["DiT training"] = probe_dit(Path(pre["output_dir"]), probe_processes, dit_steps, workdir)

    profile = assemble(ds, shapes, probes, vae_config, runs_per_campaign)
    render(profile)

    if json_out:
        json_out.write_text(json.dumps(profile, indent=2, default=str))
        console.print(f"\n[green]Wrote[/green] {json_out}")


if __name__ == "__main__":
    app()
