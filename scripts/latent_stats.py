#!/usr/bin/env python
"""VAE latent-space mean/std shift diagnostic. Not a training run.

PhD-advisor-requested (2026-08-10): quantify how far the VAE's latent
distribution moves once the shockwave fields pass through the encoder, on
two separate axes:

  1. RAW-DATA -> LATENT shift: the 4 physical channels (density,
     momentum_x, momentum_y, pressure) have very different physical
     scales; this reports what the encoder does to the (already z-scored)
     input once it lands in the C-channel latent space.
  2. PRETRAINED -> FINETUNED shift: `AutoencoderKLLTXVideo` ships with
     `latents_mean`/`latents_std` buffers baked in from its own (video)
     pretraining -- the reference distribution every downstream consumer
     (this project's `_encode`/`_decode` rescale, and eventually the DiT
     stage) assumes the latents follow. This compares that reference
     against (a) what a freshly-inflated, NEVER-finetuned encoder actually
     produces on this project's data, and (b) what the current finetuned
     checkpoint produces -- i.e. how much finetuning has pulled the latent
     distribution toward or away from the assumption baked into
     `latents_mean`/`latents_std`.

"Pretrained" here means: `load_vae(cfg.model.model_source)` + this run's
own `inflate_vae_io_channels(..., init=cfg.adapter.inflate_init)`, but with
NO checkpoint loaded on top -- i.e. the exact starting point every
finetuning run in this ledger begins from. There is no meaningful "3-channel
pretrained VAE on RGB video" baseline to compare against directly: the
encoder cannot consume 4-channel physical fields until it has been inflated.

Motivated by the KL-divergence weight experiment's own documented gap
(EXPERIMENTS.md, Open Question 19: "no existing diagnostic reports
per-epoch latent mean/std/logvar summary statistics") -- this is that
missing diagnostic, run once against a specific checkpoint rather than
per-epoch during training.

Reports, per latent channel, both:
  - `posterior_mean`/`posterior_std` (= exp(0.5*logvar)): the RAW encoder
    output before any rescale -- directly comparable to `latents_mean`/
    `latents_std`.
  - the rescaled `latents` (== `(posterior_mean - latents_mean) * scaling_factor
    / latents_std`, exactly `VaeTrainer._encode`'s formula) -- what
    everything downstream (reconstruction, and any future DiT) actually
    consumes. If the reference stats fit this data, these should land close
    to mean 0 / std 1 per channel; the gap is the "shift" in the most
    directly interpretable form.

Usage:
    python scripts/latent_stats.py configs/finetune_vae/finetune_vae_whole_structure_baseline.yaml \\
        --checkpoint $SCRATCH/windinet/finetune_vae_outputs/sng_pvc/finetune_vae_whole_structure_baseline/checkpoints/vae_shockwave_best.safetensors \\
        --num-samples 32 \\
        --output latent_stats_whole_structure_baseline.json

Must run on a single tile / single process -- this is encode-only inference
(no gradients, no accelerate launch needed).
"""

import json
from pathlib import Path

import torch
import typer
import yaml
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader, Subset

from windinet.config import VaeTrainerConfig
from windinet.inference.model_loader import load_vae
from windinet.training.shockwave_data import ShockWaveDataset, build_shockwave_video
from windinet.utils import get_default_device
from windinet.vae_adapter import inflate_vae_io_channels, load_inflated_vae_checkpoint

console = Console()
app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)

RAW_CHANNELS = ["density", "momentum_x", "momentum_y", "pressure"]


class ChannelStats:
    """Streaming per-channel mean/std over an arbitrary-rank tensor's channel axis."""

    def __init__(self, num_channels: int, device: torch.device | str = "cpu") -> None:
        self.count = 0
        self.sum = torch.zeros(num_channels, dtype=torch.float64, device=device)
        self.sumsq = torch.zeros(num_channels, dtype=torch.float64, device=device)

    def update(self, x: torch.Tensor, channel_dim: int) -> None:
        """x: any shape with `num_channels` at `channel_dim`."""
        x = x.movedim(channel_dim, 0).reshape(x.shape[channel_dim], -1).double()
        self.count += x.shape[1]
        self.sum += x.sum(dim=1)
        self.sumsq += x.square().sum(dim=1)

    def finalize(self) -> dict[str, list[float]]:
        mean = self.sum / self.count
        var = (self.sumsq / self.count - mean.square()).clamp_min(0.0)
        return {"mean": mean.tolist(), "std": var.sqrt().tolist()}


def _build_vae(cfg: VaeTrainerConfig, checkpoint: str | None, device: torch.device):
    """Fresh-inflate the pretrained VAE, optionally loading a finetuned checkpoint on top."""
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
        meta = load_inflated_vae_checkpoint(vae, ckpt_path=checkpoint, device="cpu", dtype=torch.float32)
        console.print(f"  loaded finetuned checkpoint (epoch={meta.get('epoch', '?')}): {checkpoint}")
    for p in vae.parameters():
        p.requires_grad_(False)
    vae.eval()
    return vae.to(device)


@torch.no_grad()
def _encode_stats(
    vae,
    loader: DataLoader,
    cfg: VaeTrainerConfig,
    device: torch.device,
    raw_stats: ChannelStats | None,
) -> dict:
    sf = float(getattr(vae.config, "scaling_factor", 1.0))
    latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(device, torch.float32)
    latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(device, torch.float32)
    num_latent_channels = vae.latents_mean.numel()

    posterior_mean_stats = ChannelStats(num_latent_channels, device=device)
    posterior_std_stats = ChannelStats(num_latent_channels, device=device)
    rescaled_stats = ChannelStats(num_latent_channels, device=device)

    for batch in loader:
        video = build_shockwave_video(
            batch,
            device=device,
            channel_mean=cfg.data.channel_mean,
            channel_std=cfg.data.channel_std,
            normalization_clip=cfg.data.normalization_clip,
            channel_order=cfg.data.channel_order,
            log_transform_channels=cfg.data.log_transform_channels,
        )
        if raw_stats is not None:
            # Raw physical units, pre-normalization -- matches what
            # scripts/compute_channel_stats.py reports, computed here on
            # the same sampled sims as the latent-side stats for a clean
            # apples-to-apples comparison rather than the full-dataset scan.
            raw_fields = torch.stack([batch[name].to(device).float() for name in RAW_CHANNELS], dim=1)
            raw_stats.update(raw_fields, channel_dim=1)

        out = vae.encode(video)
        posterior_mean = out.latent_dist.mean.float()
        posterior_logvar = out.latent_dist.logvar.float()
        posterior_std = (0.5 * posterior_logvar).exp()
        rescaled = (posterior_mean - latents_mean) * sf / latents_std

        posterior_mean_stats.update(posterior_mean, channel_dim=1)
        posterior_std_stats.update(posterior_std, channel_dim=1)
        rescaled_stats.update(rescaled, channel_dim=1)

    return {
        "posterior_mean": posterior_mean_stats.finalize(),
        "posterior_std": posterior_std_stats.finalize(),
        "rescaled_latents": rescaled_stats.finalize(),
    }


def _summarize(block: dict) -> dict[str, float]:
    rescaled_mean = torch.tensor(block["rescaled_latents"]["mean"])
    rescaled_std = torch.tensor(block["rescaled_latents"]["std"])
    return {
        "rescaled_mean_abs_avg": rescaled_mean.abs().mean().item(),
        "rescaled_mean_abs_max": rescaled_mean.abs().max().item(),
        "rescaled_std_avg": rescaled_std.mean().item(),
        "rescaled_std_min": rescaled_std.min().item(),
        "rescaled_std_max": rescaled_std.max().item(),
    }


@app.command()
def main(
    config_path: str = typer.Argument(..., help="VaeTrainerConfig-shaped YAML (data_root/adapter/normalization only matter)"),
    checkpoint: str = typer.Option(..., help="Finetuned inflate-mode checkpoint (ltx-inflated-io-v1 safetensors)"),
    num_samples: int = typer.Option(32, help="How many held-out (eval-split) sims to encode"),
    batch_size: int = typer.Option(4, help="Sims per encode() call"),
    split: str = typer.Option("eval", help="'eval' (held-out, never seen by the finetuned checkpoint) or 'train'"),
    output: str = typer.Option("latent_stats.json", help="Where to write the full per-channel JSON report"),
) -> None:
    with open(config_path) as f:
        cfg = VaeTrainerConfig(**yaml.safe_load(f))

    device = get_default_device()
    console.print(f"Device: {device}")

    # Reproduce VaeTrainer.train()'s exact split (same seed, same perm) so
    # "eval" here is the same held-out set the finetuned checkpoint never
    # trained on -- results aren't inflated by memorization.
    full_dataset = ShockWaveDataset(Path(cfg.data.data_root), num_sim_frames=cfg.data.num_sim_frames)
    split_generator = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(len(full_dataset), generator=split_generator).tolist()
    n_eval = min(cfg.data.eval_sims, len(full_dataset))
    n_train = len(full_dataset) - n_eval
    eval_indices = perm[n_train:n_train + n_eval]
    train_indices = perm[:n_train]
    chosen = (eval_indices if split == "eval" else train_indices)[:num_samples]
    console.print(f"Encoding {len(chosen)} sims from the '{split}' split (of {len(full_dataset)} total)")

    loader = DataLoader(
        Subset(full_dataset, chosen),
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.data.num_dataloader_workers,
    )

    console.print("\n[bold]Pretrained (freshly-inflated, no finetuning)[/bold]")
    vae_pretrained = _build_vae(cfg, checkpoint=None, device=device)
    raw_stats = ChannelStats(len(RAW_CHANNELS), device=device)
    pretrained_result = _encode_stats(vae_pretrained, loader, cfg, device, raw_stats=raw_stats)
    raw_result = raw_stats.finalize()
    reference_mean = vae_pretrained.latents_mean.tolist()
    reference_std = vae_pretrained.latents_std.tolist()
    del vae_pretrained
    if device.type != "cpu":
        getattr(torch, device.type).empty_cache()

    console.print("\n[bold]Finetuned[/bold]")
    vae_finetuned = _build_vae(cfg, checkpoint=checkpoint, device=device)
    finetuned_result = _encode_stats(vae_finetuned, loader, cfg, device, raw_stats=None)
    del vae_finetuned

    report = {
        "config": config_path,
        "checkpoint": checkpoint,
        "split": split,
        "sim_indices": chosen,
        "num_latent_channels": len(reference_mean),
        "raw_channels": {name: {"mean": raw_result["mean"][i], "std": raw_result["std"][i]} for i, name in enumerate(RAW_CHANNELS)},
        "reference_latents_mean": reference_mean,
        "reference_latents_std": reference_std,
        "pretrained": pretrained_result,
        "finetuned": finetuned_result,
    }
    Path(output).write_text(json.dumps(report, indent=2))
    console.print(f"\nFull per-channel report written to {output}")

    raw_table = Table(title="Raw physical channels (this sample, not the full dataset)")
    raw_table.add_column("channel")
    raw_table.add_column("mean", justify="right")
    raw_table.add_column("std", justify="right")
    for name in RAW_CHANNELS:
        raw_table.add_row(name, f"{report['raw_channels'][name]['mean']:.4f}", f"{report['raw_channels'][name]['std']:.4f}")
    console.print(raw_table)

    summary_table = Table(title="Rescaled latents (target: mean 0, std 1 per channel)")
    summary_table.add_column("encoder")
    summary_table.add_column("|mean| avg", justify="right")
    summary_table.add_column("|mean| max", justify="right")
    summary_table.add_column("std avg", justify="right")
    summary_table.add_column("std min", justify="right")
    summary_table.add_column("std max", justify="right")
    for label, block in [("pretrained (no finetune)", pretrained_result), ("finetuned", finetuned_result)]:
        s = _summarize(block)
        summary_table.add_row(
            label,
            f"{s['rescaled_mean_abs_avg']:.4f}",
            f"{s['rescaled_mean_abs_max']:.4f}",
            f"{s['rescaled_std_avg']:.4f}",
            f"{s['rescaled_std_min']:.4f}",
            f"{s['rescaled_std_max']:.4f}",
        )
    console.print(summary_table)

    pre_mean = torch.tensor(pretrained_result["rescaled_latents"]["mean"])
    ft_mean = torch.tensor(finetuned_result["rescaled_latents"]["mean"])
    pre_std = torch.tensor(pretrained_result["rescaled_latents"]["std"])
    ft_std = torch.tensor(finetuned_result["rescaled_latents"]["std"])
    console.print(
        "\nFinetuning-induced shift in rescaled-latent mean (L2 over channels): "
        f"{(ft_mean - pre_mean).norm().item():.4f}"
    )
    console.print(
        f"Finetuning-induced std ratio (finetuned/pretrained, avg over channels): "
        f"{(ft_std / pre_std).mean().item():.4f}"
    )
    top_shift = (ft_mean - pre_mean).abs().topk(min(5, len(ft_mean)))
    console.print(f"Top-5 latent channels by |mean shift|: {[(i.item(), round(v.item(), 4)) for v, i in zip(*top_shift)]}")


if __name__ == "__main__":
    app()
