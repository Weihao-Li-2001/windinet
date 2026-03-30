"""
Extract scalar conditioning values from meta.json files and save them for training.

Special scalar handling:
- inlet_speed_mps: extracted from 'wind_speed_mps' field in meta.json
- field_size_m: computed as 'city_diameter_m' + 600 (600m padding around the city)
"""

import json
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from windinet.utils import logger


def load_dataset_entries(dataset_file: Path) -> list[dict[str, Any]]:
    """Load entries from dataset file (CSV/JSON/JSONL)."""
    if dataset_file.suffix == ".json":
        with open(dataset_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")
        return data
    elif dataset_file.suffix == ".jsonl":
        entries = []
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                entries.append(json.loads(line))
        return entries
    elif dataset_file.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(dataset_file)
        return df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_file.suffix}")


def extract_scalars_from_meta(
    meta_path: Path,
    scalar_names: list[str],
) -> dict[str, float]:
    """Extract scalar values from a meta.json file."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if isinstance(meta, list):
        if len(meta) == 0:
            raise ValueError(f"Empty meta.json list in {meta_path}")
        meta = meta[0]

    scalars = {}
    for name in scalar_names:
        if name == "inlet_speed_mps":
            if "wind_speed_mps" not in meta:
                raise KeyError(f"'wind_speed_mps' not found in {meta_path}")
            scalars[name] = float(meta["wind_speed_mps"])
        elif name == "field_size_m":
            if "field_size_m" in meta:
                scalars[name] = float(meta["field_size_m"])
            elif "city_diameter_m" in meta:
                # field_size_m = city_diameter_m + 600m padding
                scalars[name] = float(meta["city_diameter_m"]) + 600.0
            else:
                raise KeyError(f"Neither 'field_size_m' nor 'city_diameter_m' found in {meta_path}")
        else:
            if name not in meta:
                raise KeyError(f"'{name}' not found in {meta_path}")
            scalars[name] = float(meta[name])

    return scalars


def compute_scalars(
    dataset_file: str | Path,
    output_dir: str,
    media_column: str = "media_path",
    scalar_names: list[str] | None = None,
    meta_filename: str = "meta.json",
    meta_root: str | None = None,
    split: str = "train",
) -> None:
    """Extract scalar values from meta.json files and save them as tensors."""
    console = Console()

    if scalar_names is None:
        scalar_names = ["inlet_speed_mps", "field_size_m"]

    dataset_file = Path(dataset_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    entries = load_dataset_entries(dataset_file)
    logger.info(f"Loaded {len(entries):,} entries from dataset")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting scalars", total=len(entries))

        processed = 0
        skipped = 0

        for entry in entries:
            if media_column not in entry:
                raise ValueError(f"Column '{media_column}' not found in entry: {entry}")

            media_path = Path(entry[media_column].strip())

            if meta_root:
                sample_id = media_path.stem
                meta_path = Path(meta_root) / split / sample_id / meta_filename
            else:
                meta_path = media_path.parent / meta_filename

            if not meta_path.exists():
                logger.warning(f"Meta file not found: {meta_path}")
                skipped += 1
                progress.advance(task)
                continue

            try:
                scalars = extract_scalars_from_meta(meta_path, scalar_names)

                scalar_tensor = torch.tensor(
                    [scalars[name] for name in scalar_names],
                    dtype=torch.float32,
                )

                output_rel_path = media_path.with_suffix(".pt")
                output_file = output_path / output_rel_path
                output_file.parent.mkdir(parents=True, exist_ok=True)

                scalar_data = {
                    "scalars": scalar_tensor,
                    "scalar_names": scalar_names,
                }
                torch.save(scalar_data, output_file)
                processed += 1

            except Exception as e:
                logger.warning(f"Failed to process {media_path}: {e}")
                skipped += 1

            progress.advance(task)

    logger.info(f"Processed {processed:,} samples, skipped {skipped:,}")
    logger.info(f"Scalar tensors saved to {output_path}")
