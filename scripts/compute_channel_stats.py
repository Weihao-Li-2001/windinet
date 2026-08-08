"""
Compute per-channel mean/std/min/max over an entire ShockWave HDF5 dataset.

Motivation: `configs/finetune_vae/euler_mq_128_only_train.yaml` carries a
hand-computed `data_normalization_stats` block (density/momentum_x/
momentum_y/pressure/log_density -- no log_pressure) that nothing in the
codebase actually loads or references (it isn't a valid VaeTrainerConfig,
just a standalone notes file) -- there's no way to tell whether it's still
accurate, was computed on the same data cut, or used the same std
convention as the rest of this codebase without recomputing it directly.
This script does that recomputation, plus the log_pressure that file never
had, from the real HDF5 file.

This project's "train"/"val" split (VaeDataConfig.eval_sims) is a runtime
random split of ONE HDF5 file via torch.Generator, not separate files --
so processing every simulation in the given --h5 path already covers both
halves; there's no separate val.h5 to also read.

Streams one simulation at a time (running sum/sum-of-squares/min/max)
rather than concatenating the whole dataset into memory, so this scales to
the 256x256_ds / 512x512_orig resolutions too, not just 128x128_ds
(4x / 16x the pixels per frame respectively) -- pass any of the three via
--h5 unchanged.

Usage:
    python scripts/compute_channel_stats.py euler_mq_dataset/128x128_ds/train.h5
    python scripts/compute_channel_stats.py euler_mq_dataset/256x256_ds/train.h5 \
        --output euler_mq_dataset/256x256_ds/channel_stats.json
"""

import json
import math
from pathlib import Path

import h5py
import numpy as np
import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

console = Console()
app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)

RAW_CHANNELS = ["density", "momentum_x", "momentum_y", "pressure"]
LOG_CHANNELS = {"log_density": "density", "log_pressure": "pressure"}
ALL_STATS_NAMES = RAW_CHANNELS + list(LOG_CHANNELS)


class RunningStats:
    """Streaming mean/std/min/max (population std, ddof=0 -- matches this
    codebase's variance convention elsewhere, e.g. windinet.losses.vrms's
    `unbiased=False`), accumulated one simulation's worth of samples at a time.
    """

    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = math.inf
        self.max = -math.inf

    def update(self, values: np.ndarray) -> None:
        flat = values.reshape(-1).astype(np.float64)
        self.count += flat.size
        self.sum += float(flat.sum())
        self.sumsq += float(np.square(flat).sum())
        self.min = min(self.min, float(flat.min()))
        self.max = max(self.max, float(flat.max()))

    def finalize(self) -> dict[str, float]:
        mean = self.sum / self.count
        variance = self.sumsq / self.count - mean**2
        return {
            "mean": mean,
            "std": math.sqrt(max(variance, 0.0)),
            "min": self.min,
            "max": self.max,
            "count": self.count,
        }


@app.command()
def main(
    h5_path: Path = typer.Argument(..., help="Path to the ShockWave HDF5 file (e.g. .../128x128_ds/train.h5)"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write results as JSON here, shaped like euler_mq_128_only_train.yaml's data_normalization_stats"
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help=(
            "Only process a random sample of N simulations (quick look, e.g. on a "
            "not-yet-fully-downloaded 256x256_ds) -- NOT the first N by sorted id, "
            "since sample ids may be grouped by gamma/generation batch and an "
            "unshuffled prefix can silently be a biased, non-representative slice"
        ),
    ),
    limit_seed: int = typer.Option(42, "--limit-seed", help="Seed for the --limit random sample, for reproducibility"),
    compare_to: Path | None = typer.Option(
        None,
        "--compare-to",
        help="A YAML file with a data_normalization_stats block (e.g. configs/finetune_vae/euler_mq_128_only_train.yaml) to diff the computed stats against",
    ),
) -> None:
    """Stream every simulation in --h5 and report mean/std/min/max per channel."""
    import os

    os.environ["HDF5_EXT_PREFIX"] = str(h5_path.parent)
    with h5py.File(h5_path, "r") as f:
        ids = sorted(f.keys())
        if limit is not None:
            # Random sample, not ids[:limit] -- sample ids may be grouped by
            # gamma/generation batch, so an unshuffled prefix can silently be
            # a biased, non-representative slice (seen in practice: a --limit
            # 50 run landed ~15-35% high on std across every raw channel).
            rng = np.random.default_rng(limit_seed)
            ids = list(rng.choice(ids, size=min(limit, len(ids)), replace=False))
        console.print(f"[cyan]{len(ids)} simulations in {h5_path}[/cyan]")

        stats = {name: RunningStats() for name in ALL_STATS_NAMES}

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning simulations", total=len(ids))
            for sid in ids:
                group = f[sid]
                raw_values = {}
                for name in RAW_CHANNELS:
                    values = group[name][:]
                    raw_values[name] = values
                    stats[name].update(values)
                for log_name, source in LOG_CHANNELS.items():
                    source_values = raw_values[source]
                    if np.any(source_values <= 0):
                        raise ValueError(
                            f"{sid}/{source} has non-positive values (min={source_values.min()}); "
                            f"log({source}) is undefined there"
                        )
                    stats[log_name].update(np.log(source_values))
                progress.advance(task)

    results = {name: stat.finalize() for name, stat in stats.items()}

    table = Table(title=f"Channel statistics -- {h5_path}")
    for col in ("channel", "mean", "std", "min", "max", "count"):
        table.add_column(col)
    for name in ALL_STATS_NAMES:
        r = results[name]
        table.add_row(name, f"{r['mean']:.6f}", f"{r['std']:.6f}", f"{r['min']:.6f}", f"{r['max']:.6f}", f"{r['count']:,}")
    console.print(table)

    if compare_to is not None:
        import yaml

        with open(compare_to) as fh:
            reference = yaml.safe_load(fh).get("data_normalization_stats", {})
        diff_table = Table(title=f"Diff vs {compare_to}")
        for col in ("channel", "stat", "computed", "reference", "abs diff", "rel diff"):
            diff_table.add_column(col)
        for name in ALL_STATS_NAMES:
            ref = reference.get(name)
            if ref is None:
                diff_table.add_row(name, "-", "-", "MISSING", "-", "-")
                continue
            for stat_name in ("mean", "std", "min", "max"):
                computed = results[name][stat_name]
                reference_value = ref[stat_name]
                abs_diff = abs(computed - reference_value)
                rel_diff = abs_diff / abs(reference_value) if reference_value != 0 else float("inf")
                diff_table.add_row(
                    name, stat_name, f"{computed:.6f}", f"{reference_value:.6f}", f"{abs_diff:.2e}", f"{rel_diff:.2%}"
                )
        console.print(diff_table)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as fh:
            json.dump(
                {
                    "source_h5": str(h5_path),
                    "num_simulations": len(ids),
                    "data_normalization_stats": {
                        name: {k: v for k, v in r.items() if k != "count"} for name, r in results.items()
                    },
                },
                fh,
                indent=2,
            )
        console.print(f"[green]Wrote {output}[/green]")


if __name__ == "__main__":
    app()
