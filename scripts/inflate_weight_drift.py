#!/usr/bin/env python
"""VAE inflate-init weight-drift diagnostic. Not a training run.

PhD-advisor-motivated (2026-08-12): the whole-structure baseline's 4th
channel (the freshly-grown slot, `inflate_init: mean`) has low val_vrmse but
visually still looks like a superposition of the 3 pretrained channels --
reconstructions for it resemble a blend of the other fields rather than a
clean, independent prediction. Two competing explanations:

  1. Training hasn't run long enough to pull the fresh channel's weights
     away from their `mean`-of-the-other-3 starting point.
  2. The weights HAVE moved, but a mostly-RMSE loss on physically-correlated
     fields (density/momentum/pressure share the same shock structure) keeps
     landing near a solution that still resembles an average -- more epochs
     wouldn't fix this, only a different init/loss would.

This script answers "did the weights move" directly, with no forward pass,
no data loading, no GPU -- pure weight-space comparison between the
freshly-inflated (pretrained, no finetuning) VAE and a finetuned checkpoint:

  - `own_drift`: per-channel cosine similarity + relative L2 change between
    a channel's weight block at init vs after finetuning. Low similarity /
    high relative change = that channel's weights moved a lot.
  - `fresh_vs_mean_of_others`: cosine similarity between the fresh channel's
    weight block and the mean of the (still-pretrained-derived) original 3
    channels' weight blocks, computed separately at init and after
    finetuning. At init this is ~1.0 by construction (`inflate_init: mean`
    IS that average). If it stays high after finetuning, the fresh
    channel's decoder projection still statistically resembles "the average
    of the other 3" even though its own weights changed (own_drift can be
    high AND fresh_vs_mean_of_others can stay high at the same time -- e.g.
    if the other 3 channels' weights also drifted together with it).

Read together: high own_drift + low fresh_vs_mean_of_others after
finetuning means training successfully specialized the fresh channel --
if the visual artifact still persists in that case, more training is
unlikely to be the fix. Low own_drift means the weights are still close to
their init regardless of anything else -- that DOES point at "insufficient
training" (or a too-low LR on that slot, see `encoder_tail_lr_multiplier`
-- note `decoder.conv_out` is not scaled by that multiplier, it trains at
the full decoder LR).

Checks both `encoder.conv_in` (input side, only informative if
`freeze_conv_in: false`) and `decoder.conv_out` (output side -- what
directly produces the visually-inspected reconstruction).

Usage:
    python scripts/inflate_weight_drift.py \\
        configs/finetune_vae/finetune_vae_whole_structure_baseline.yaml \\
        --checkpoint $SCRATCH/windinet/finetune_vae_outputs/sng_pvc/finetune_vae_whole_structure_baseline/checkpoints/vae_shockwave_best.safetensors \\
        --output weight_drift_whole_structure_baseline.json

Runs on CPU by default -- this is a handful of small conv weight tensors,
not worth requesting a GPU node for.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
import typer
import yaml
from rich.console import Console
from rich.table import Table

from windinet.config import VaeTrainerConfig
from windinet.inference.model_loader import load_vae
from windinet.vae_adapter import inflate_vae_io_channels, load_inflated_vae_checkpoint

console = Console()
app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)

# LTX-Video's native channel count -- indices [0, N_ORIG) keep the pretrained
# RGB conv weights byte-for-byte at init; index N_ORIG onward is what
# inflate_vae_io_channels grows fresh (windinet/vae_adapter.py:220-343).
N_ORIG = 3


def _build_vae(cfg: VaeTrainerConfig, checkpoint: str | None):
    vae = load_vae(cfg.model.model_source, dtype=torch.float32)
    adapter_cfg = cfg.adapter
    n = len(adapter_cfg.channels)
    copy_from_index = (
        adapter_cfg.channels.index(adapter_cfg.inflate_copy_channel)
        if adapter_cfg.inflate_init == "copy"
        else None
    )
    inflate_vae_io_channels(vae, n=n, init=adapter_cfg.inflate_init, copy_from_index=copy_from_index)
    if checkpoint is not None:
        load_inflated_vae_checkpoint(vae, ckpt_path=checkpoint, device="cpu", dtype=torch.float32)
    return vae


def _per_channel_blocks(weight: torch.Tensor, n: int, axis: int) -> torch.Tensor:
    """weight with `n*block` along `axis` -> [n, ...] with block folded into axis+1."""
    w = weight.movedim(axis, 0)
    block = w.shape[0] // n
    return w.reshape(n, block, *w.shape[1:]).flatten(1).double()


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def _analyze(name: str, w_init: torch.Tensor, w_finetuned: torch.Tensor, n: int, axis: int, channels: list[str]) -> dict:
    blocks_init = _per_channel_blocks(w_init, n, axis)
    blocks_ft = _per_channel_blocks(w_finetuned, n, axis)

    own_drift = {}
    for c in range(n):
        cos = _cos(blocks_init[c], blocks_ft[c])
        rel_l2 = ((blocks_ft[c] - blocks_init[c]).norm() / blocks_init[c].norm().clamp_min(1e-12)).item()
        own_drift[channels[c]] = {"cosine_sim_to_init": cos, "relative_l2_change": rel_l2}

    fresh_vs_mean = {}
    for c in range(N_ORIG, n):
        mean_init = blocks_init[:N_ORIG].mean(dim=0)
        mean_ft = blocks_ft[:N_ORIG].mean(dim=0)
        fresh_vs_mean[channels[c]] = {
            "at_init": _cos(blocks_init[c], mean_init),
            "after_finetuning": _cos(blocks_ft[c], mean_ft),
        }

    return {"layer": name, "own_drift": own_drift, "fresh_vs_mean_of_others": fresh_vs_mean}


@app.command()
def main(
    config_path: str = typer.Argument(..., help="VaeTrainerConfig-shaped YAML (only adapter.channels/inflate_init matter)"),
    checkpoint: str = typer.Option(..., help="Finetuned inflate-mode checkpoint (ltx-inflated-io-v1 safetensors)"),
    output: str = typer.Option("weight_drift.json", help="Where to write the full per-channel JSON report"),
) -> None:
    with open(config_path) as f:
        cfg = VaeTrainerConfig(**yaml.safe_load(f))
    channels = list(cfg.adapter.channels)
    n = len(channels)

    console.print(f"channels={channels} (fresh slot(s): {channels[N_ORIG:]}) inflate_init={cfg.adapter.inflate_init!r}")

    vae_pretrained = _build_vae(cfg, checkpoint=None)
    vae_finetuned = _build_vae(cfg, checkpoint=checkpoint)

    report = {
        "config": config_path,
        "checkpoint": checkpoint,
        "channels": channels,
        "inflate_init": cfg.adapter.inflate_init,
        "encoder_conv_in": _analyze(
            "encoder.conv_in",
            vae_pretrained.encoder.conv_in.conv.weight,
            vae_finetuned.encoder.conv_in.conv.weight,
            n,
            axis=1,
            channels=channels,
        ),
        "decoder_conv_out": _analyze(
            "decoder.conv_out",
            vae_pretrained.decoder.conv_out.conv.weight,
            vae_finetuned.decoder.conv_out.conv.weight,
            n,
            axis=0,
            channels=channels,
        ),
    }
    Path(output).write_text(json.dumps(report, indent=2))
    console.print(f"\nFull per-channel report written to {output}")

    for block in (report["encoder_conv_in"], report["decoder_conv_out"]):
        table = Table(title=f"{block['layer']} -- own weight drift (init -> finetuned)")
        table.add_column("channel")
        table.add_column("cosine sim to init", justify="right")
        table.add_column("relative L2 change", justify="right")
        for ch, stats in block["own_drift"].items():
            table.add_row(ch, f"{stats['cosine_sim_to_init']:.4f}", f"{stats['relative_l2_change']:.4f}")
        console.print(table)

        if block["fresh_vs_mean_of_others"]:
            table2 = Table(title=f"{block['layer']} -- fresh channel vs mean(original {N_ORIG})")
            table2.add_column("channel")
            table2.add_column("cosine sim at init", justify="right")
            table2.add_column("cosine sim after finetuning", justify="right")
            for ch, stats in block["fresh_vs_mean_of_others"].items():
                table2.add_row(ch, f"{stats['at_init']:.4f}", f"{stats['after_finetuning']:.4f}")
            console.print(table2)


if __name__ == "__main__":
    app()
