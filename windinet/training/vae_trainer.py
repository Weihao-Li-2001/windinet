# Based on finetune_decoder.py from the WinDiNet development codebase.
# Refactored to use Accelerate for DDP, pydantic configs, and the windinet package.

"""Shockwave CFD VAE finetuning with trainable channel adapters.

The pretrained LTX-Video encoder is frozen. The decoder and the 4->3 / 3->4
adapters are trained using reconstruction, gradient, structural, and wavelet
losses on normalized HDF5 simulation fields.
"""

import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import rich
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from pydantic import BaseModel
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Subset

from windinet.config import VaeTrainerConfig
from windinet.inference.model_loader import load_vae
from windinet.vae_adapter import AdaptedVAE, load_adapted_vae

from windinet.losses import (
    reconstruction_losses,
    SSIMLoss,
)

from windinet.loss_weighting import (
    build_loss_weighting,
    GradNorm,
)

from windinet.loss_weighting.utils import (
    compute_grad_norms,
)

from windinet.training.shockwave_data import ShockWaveDataset, build_shockwave_video
from windinet.utils import logger

IS_MAIN_PROCESS = os.environ.get("LOCAL_RANK", "0") == "0"


@torch.no_grad()
def vrmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Variance-normalized RMSE, averaged over the batch."""
    diff = (pred - target).float()
    mse = diff.square().mean(dim=(1, 2, 3, 4))
    variance = target.float().var(dim=(1, 2, 3, 4), unbiased=False)
    return float(torch.sqrt(mse / (variance + eps)).mean().item())


class VaeTrainer:
    def __init__(self, config: VaeTrainerConfig) -> None:
        self._config = config
        self._print_config(config)
        self._setup_accelerator()
        self._load_vae()
        self._collect_trainable_params()
        self._init_wandb()

        self.ssim_loss = SSIMLoss(
            channels=4,
            window_size=11,
            sigma=1.5,
        ).to(self._accelerator.device)


        self.loss_weighter = build_loss_weighting(
            config.loss_weighting
        )

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def _load_vae(self) -> None:
        """Load the VAE, freeze encoder, and unfreeze decoder/adapters."""
        self._vae = load_vae(self._config.model.model_source, dtype=torch.float32)

        adapter_cfg = self._config.adapter
        if adapter_cfg.enabled:
            self._vae, meta = load_adapted_vae(
                self._vae,
                ckpt_path=adapter_cfg.checkpoint,
                device="cpu",
                dtype=torch.float32,
                channels=adapter_cfg.channels,
                k=adapter_cfg.hidden_channels,
                activation=adapter_cfg.activation,
                default_temb=adapter_cfg.default_temb,
            )
            logger.info(
                "VAE adapters enabled: "
                f"channels={meta['channels']}, hidden_channels={meta['k']}, "
                f"activation={meta['activation']}"
            )
            self._adapter_meta = meta
        else:
            self._adapter_meta = None

        # Freeze everything
        for p in self._vae.parameters():
            p.requires_grad_(False)

        # Unfreeze decoder and, when enabled, both channel adapters.
        decoder = self._get_decoder()
        for p in decoder.parameters():
            p.requires_grad_(True)

        if isinstance(self._vae, AdaptedVAE):
            self._vae.in_adapter.requires_grad_(True)
            self._vae.out_adapter.requires_grad_(True)

        if self._config.optimization.enable_gradient_checkpointing:
            base_vae = self._vae.vae if isinstance(self._vae, AdaptedVAE) else self._vae
            base_vae.enable_gradient_checkpointing()
            logger.info("VAE gradient checkpointing enabled")

        # Keep the frozen encoder deterministic; only explicitly trainable
        # modules are switched back to training mode in the epoch loop.
        self._vae.eval()

        logger.info(f"VAE loaded. Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")

    def _unwrap_vae(self) -> nn.Module:
        return self._accelerator.unwrap_model(self._vae)

    def _get_decoder(self) -> nn.Module:
        vae = self._unwrap_vae()
        if isinstance(vae, AdaptedVAE):
            return vae.vae.decoder
        for name in ("decoder", "vae_decoder"):
            if hasattr(vae, name) and isinstance(getattr(vae, name), nn.Module):
                return getattr(vae, name)
        return vae

    def _collect_trainable_params(self) -> None:
        self._trainable_params = [p for p in self._vae.parameters() if p.requires_grad]
        logger.debug(f"Trainable params: {sum(p.numel() for p in self._trainable_params):,}")

    def _set_trainable_modules_mode(self, training: bool) -> None:
        self._get_decoder().train(training)
        vae = self._unwrap_vae()
        if isinstance(vae, AdaptedVAE):
            vae.in_adapter.train(training)
            vae.out_adapter.train(training)

    # ------------------------------------------------------------------
    # VAE encode / decode
    # ------------------------------------------------------------------

    def _encode(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video to normalised latents. video: [B, C, F, H, W]."""
        out = self._vae.encode(video)
        latents = out.latent_dist.mean
        mean = self._vae.latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        std = self._vae.latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        sf = float(getattr(self._vae.config, "scaling_factor", 1.0))
        return (latents - mean) * sf / std

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode normalised latents to video."""
        mean = self._vae.latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        std = self._vae.latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        sf = float(getattr(self._vae.config, "scaling_factor", 1.0))
        z = latents * std / sf + mean
        temb = torch.full(
            (z.shape[0],),
            self._config.adapter.default_temb,
            device=z.device,
            dtype=z.dtype,
        )
        return self._vae.decode(z, temb=temb, return_dict=True).sample

    def _forward_pass(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Encode → decode through the VAE. Returns (reconstruction, original_frames)."""
        orig_F = x.shape[2]
        latents = self._encode(x)
        recon = self._decode(latents)
        return recon[:, :, :orig_F], orig_F

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> Path:
        """Run the VAE decoder finetuning loop."""
        cfg = self._config
        device = self._accelerator.device
        set_seed(cfg.seed)

        # Data
        full_dataset = ShockWaveDataset(
            cfg.data.data_root,
            num_sim_frames=cfg.data.num_sim_frames,
        )
        n_eval = min(cfg.data.eval_sims, len(full_dataset))
        n_train = len(full_dataset) - n_eval
        if n_train < 1 or n_eval < 1:
            raise ValueError(
                f"Need at least one training and one evaluation sample; got {n_train} train, {n_eval} eval"
            )
        train_set = Subset(full_dataset, list(range(n_train)))
        eval_set = Subset(full_dataset, list(range(n_train, n_train + n_eval)))

        train_loader = DataLoader(
            train_set,
            batch_size=cfg.optimization.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_dataloader_workers,
            drop_last=True,
        )
        eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False)

        logger.info(f"Dataset: {n_train} train, {n_eval} eval samples")

        # Optimizer + scheduler
        optimizer = torch.optim.AdamW(
            self._trainable_params,
            lr=cfg.optimization.learning_rate,
            weight_decay=cfg.optimization.weight_decay,
        )

        steps_per_epoch = math.ceil(len(train_loader) / cfg.optimization.gradient_accumulation_steps)
        total_opt_steps = max(1, cfg.optimization.epochs * steps_per_epoch)
        warmup_steps = min(cfg.optimization.warmup_steps, total_opt_steps)

        if warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=cfg.optimization.warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_opt_steps - warmup_steps),
                eta_min=cfg.optimization.min_learning_rate,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_opt_steps, eta_min=cfg.optimization.min_learning_rate,
            )

        # Prepare with Accelerate
        self._vae = self._accelerator.prepare(self._vae)
        optimizer = self._accelerator.prepare(optimizer)
        train_loader = self._accelerator.prepare(train_loader)

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self._save_config()

        # Progress bar
        if IS_MAIN_PROCESS:
            train_progress = Progress(
                TextColumn("Epoch {task.fields[epoch]}"),
                BarColumn(bar_width=40, style="blue"),
                MofNCompleteColumn(),
                TextColumn("Loss: {task.fields[loss]:.4f}"),
                TextColumn("LR: {task.fields[lr]:.2e}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(compact=True),
            )
            live = Live(Panel(train_progress), refresh_per_second=2)
        else:
            train_progress = MagicMock()
            live = nullcontext()

        global_opt_step = 0
        saved_path = None

        logger.info("Starting VAE decoder finetuning...")

        with live:
            for epoch in range(1, cfg.optimization.epochs + 1):
                self._set_trainable_modules_mode(True)
                running_loss = 0.0
                count = 0

                loss_sum = {
                    "rmse":0.0,
                    "h1":0.0,
                    "ssim":0.0,
                    "mlw":0.0,
                }

                grad_norm_sum = {
                    "rmse":0.0,
                    "h1":0.0,
                    "ssim":0.0,
                    "mlw":0.0,
                }

                task = train_progress.add_task(
                    f"Epoch {epoch}", total=len(train_loader),
                    epoch=epoch, loss=0.0, lr=cfg.optimization.learning_rate,
                )

                optimizer.zero_grad(set_to_none=True)

                for i, batch in enumerate(train_loader):
                    orig_F = batch["density"].shape[1]
                    x = build_shockwave_video(
                        batch,
                        device=device,
                        channel_mean=cfg.data.channel_mean,
                        channel_std=cfg.data.channel_std,
                        normalization_clip=cfg.data.normalization_clip,
                    )
                    recon, _ = self._forward_pass(x)
                    x_target = x[:, :, :orig_F]
                    recon = recon[:, :, :orig_F]

                    losses = reconstruction_losses(
                        pred=recon,
                        target=x_target,
                        ssim_module=self.ssim_loss,
                        wavelet=cfg.loss.wavelet,
                        spatial_level=cfg.loss.spatial_level,
                        temporal_level=cfg.loss.temporal_level,
                        mlw_beta=cfg.loss.mlw_beta,
                        mlw_eps=cfg.loss.mlw_eps,
                    )

                    grad_norms = None


                    if isinstance(
                        self.loss_weighter,
                        GradNorm
                    ):

                        grad_norms = compute_grad_norms(
                            losses=losses,
                            parameters=self._trainable_params,
                        )
                    
                    for k,v in losses.items():
                        loss_sum[k] += v.item()

                    if grad_norms is not None:

                        for k,v in grad_norms.items():

                            grad_norm_sum[k] += v

                    weights = self.loss_weighter.get_weights()


                    total_loss = sum(
                        weights[name] * value
                        for name, value in losses.items()
                    )

                    # Accelerator.backward already divides by its configured
                    # gradient_accumulation_steps. Compensate only for a final
                    # short accumulation group (e.g. 5 batches with grad_acc=4).
                    grad_acc = cfg.optimization.gradient_accumulation_steps
                    group_start = (i // grad_acc) * grad_acc
                    group_size = min(grad_acc, len(train_loader) - group_start)
                    backward_loss = total_loss * (grad_acc / group_size)
                    self._accelerator.backward(backward_loss)

                    running_loss += total_loss.item()
                    count += 1

                    do_step = ((i + 1) % cfg.optimization.gradient_accumulation_steps == 0) or (i == len(train_loader) - 1)
                    if do_step:
                        if cfg.optimization.max_grad_norm > 0:
                            self._accelerator.clip_grad_norm_(self._trainable_params, cfg.optimization.max_grad_norm)

                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_opt_step += 1

                        # Visualization: GT | Reconstruction | Error video
                        if (
                            IS_MAIN_PROCESS
                            and cfg.optimization.vis_every_steps > 0
                            and global_opt_step % cfg.optimization.vis_every_steps == 0
                        ):
                            self._save_visualization(eval_loader, device, global_opt_step)

                        if IS_MAIN_PROCESS and global_opt_step % 10 == 0:
                            lr = optimizer.param_groups[0]["lr"]
                            avg = running_loss / max(count, 1)
                            self._accelerator.print(
                                f"epoch {epoch} step {global_opt_step}  "
                                f"loss={avg:.6f}  lr={lr:.2e}"
                                f"  [rmse={losses['rmse'].item():.4f} "
                                f"h1={losses['h1'].item():.4f} "
                                f"ssim={losses['ssim'].item():.4f} "
                                f"mlw={losses['mlw'].item():.4f}]"
                            )

                    if IS_MAIN_PROCESS:
                        train_progress.update(
                            task, advance=1,
                            loss=running_loss / max(count, 1),
                            lr=optimizer.param_groups[0]["lr"],
                        )

                epoch_losses = {
                    k: v / count
                    for k,v in loss_sum.items()
                }

                epoch_grad_norms = None


                if isinstance(
                    self.loss_weighter,
                    GradNorm
                ):

                    epoch_grad_norms = {
                        k:v/max(count,1)
                        for k,v in grad_norm_sum.items()
                    }

                self.loss_weighter.update(
                    losses=epoch_losses,
                    grad_norms=epoch_grad_norms,
                )

                weights = self.loss_weighter.get_weights()

                logger.info(
                    "Loss weights: "
                    + ", ".join(
                        [
                            f"{k}={v:.4f}"
                            for k, v in weights.items()
                        ]
                    )
                )

                # End of epoch: eval + checkpoint

                train_progress.remove_task(task)
                

                if IS_MAIN_PROCESS:
                    avg_loss = running_loss / max(count, 1)
                    ev = self._eval_vrmse(eval_loader, device)
                    lr = optimizer.param_groups[0]["lr"]
                    logger.info(f"Epoch {epoch}: loss={avg_loss:.6f}  VRMSE={ev:.6f}  lr={lr:.2e}")

                    self._log_metrics({
                        "epoch": epoch,
                        "epoch/train_loss": avg_loss,
                        "epoch/eval_vrmse": ev,
                        "epoch/learning_rate": lr,
                    })

                    if cfg.checkpoints.interval and epoch % cfg.checkpoints.interval == 0:
                        saved_path = self._save_checkpoint(epoch, global_opt_step)

                self._accelerator.wait_for_everyone()

        # Final checkpoint
        if IS_MAIN_PROCESS:
            saved_path = self._save_checkpoint(cfg.optimization.epochs, global_opt_step)

        if self._wandb_run is not None:
            self._wandb_run.finish()

        self._accelerator.end_training()
        return saved_path

    @torch.no_grad()
    def _eval_vrmse(self, eval_loader: DataLoader, device: torch.device) -> float:
        self._set_trainable_modules_mode(False)
        vals = []
        for batch in eval_loader:
            orig_F = batch["density"].shape[1]
            x = build_shockwave_video(
                batch,
                device=device,
                channel_mean=self._config.data.channel_mean,
                channel_std=self._config.data.channel_std,
                normalization_clip=self._config.data.normalization_clip,
            )
            recon, _ = self._forward_pass(x)
            vals.append(vrmse(recon[:, :, :orig_F], x[:, :, :orig_F]))
        self._set_trainable_modules_mode(True)
        return sum(vals) / max(1, len(vals))

    @torch.no_grad()
    def _save_visualization(self, eval_loader: DataLoader, device: torch.device, step: int) -> None:
        """Save GT | Reconstruction | Error video for the first eval sample."""
        logger.info("Visualization not implemented for BubbleDiNet yet.")
        return
        # from windinet.visualization import render_error_video

        # decoder = self._get_decoder()
        # decoder.eval()

        # batch = next(iter(eval_loader))
        # x = build_shockwave_video(batch, device=device)
        # recon, orig_F = self._forward_pass(x)
        # x = x[:, :, :orig_F]
        # recon = recon[:, :, :orig_F]

        # # Convert from [-1, 1] normalised to m/s
        # wind_norm = self._config.data.wind_norm
        # u_gt = x[0, 0, 1:].cpu().numpy() * wind_norm  # skip conditioning frame
        # v_gt = x[0, 1, 1:].cpu().numpy() * wind_norm
        # u_pred = recon[0, 0, 1:].cpu().numpy() * wind_norm
        # v_pred = recon[0, 1, 1:].cpu().numpy() * wind_norm
        # bmask = x[0, 2, 1].cpu().numpy() < 0  # b channel: -1=building, +1=fluid

        # vis_dir = Path(self._config.output_dir) / "vis"
        # vis_dir.mkdir(parents=True, exist_ok=True)
        # path = vis_dir / f"step_{step:06d}.mp4"
        # render_error_video(u_gt, v_gt, u_pred, v_pred, bmask, path)
        # logger.info(f"Visualization saved: {path.relative_to(self._config.output_dir)}")

        # decoder.train()

    # ------------------------------------------------------------------
    # Checkpointing (safetensors, compatible with load_adapted_vae)
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, step: int) -> Path:
        save_dir = Path(self._config.output_dir) / "checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)

        vae = self._unwrap_vae()
        decoder = self._get_decoder()
        tensors = {
            f"decoder.{k}": v.detach().cpu().contiguous()
            for k, v in decoder.state_dict().items()
        }

        adapter_cfg = self._config.adapter
        if isinstance(vae, AdaptedVAE):
            tensors.update({
                f"in_adapter.{k}": v.detach().cpu().contiguous()
                for k, v in vae.in_adapter.state_dict().items()
            })
            tensors.update({
                f"out_adapter.{k}": v.detach().cpu().contiguous()
                for k, v in vae.out_adapter.state_dict().items()
            })

        metadata = {
            "format": "ltx-decoder-plus-adapters-v1",
            "channels": str(vae.channels if isinstance(vae, AdaptedVAE) else adapter_cfg.channels),
            "n": str(len(vae.channels) if isinstance(vae, AdaptedVAE) else len(adapter_cfg.channels)),
            "k": str(vae.k if isinstance(vae, AdaptedVAE) else 0),
            "activation": str(
                self._adapter_meta["activation"] if self._adapter_meta else adapter_cfg.activation
            ),
            "default_temb": str(vae.default_temb if isinstance(vae, AdaptedVAE) else adapter_cfg.default_temb),
            "normalization": "clipped_zscore",
            "channel_mean": str(self._config.data.channel_mean),
            "channel_std": str(self._config.data.channel_std),
            "normalization_clip": str(self._config.data.normalization_clip),
            "epoch": str(epoch),
            "step": str(step),
        }

        filename = f"vae_shockwave_epoch{epoch:03d}.safetensors"
        path = save_dir / filename
        save_file(tensors, path, metadata=metadata)
        logger.info(f"VAE checkpoint saved: {path.relative_to(self._config.output_dir)}")
        return path

    # ------------------------------------------------------------------
    # Accelerator, wandb, config printing
    # ------------------------------------------------------------------

    def _setup_accelerator(self) -> None:
        self._accelerator = Accelerator(
            mixed_precision=self._config.acceleration.mixed_precision_mode,
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
        )
        if self._accelerator.num_processes > 1:
            logger.info(f"Distributed training: {self._accelerator.num_processes} processes")

    def _init_wandb(self) -> None:
        if not self._config.wandb.enabled or not IS_MAIN_PROCESS:
            self._wandb_run = None
            return
        self._wandb_run = wandb.init(
            project=self._config.wandb.project,
            entity=self._config.wandb.entity,
            name=Path(self._config.output_dir).name,
            tags=self._config.wandb.tags,
            config=self._config.model_dump(),
        )

    def _log_metrics(self, metrics: dict) -> None:
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)

    def _save_config(self) -> None:
        if not IS_MAIN_PROCESS:
            return
        config_path = Path(self._config.output_dir) / "training_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self._config.model_dump(), f, default_flow_style=False, indent=2)
        logger.info(f"Config saved: {config_path}")

    @staticmethod
    def _print_config(config: BaseModel) -> None:
        if not IS_MAIN_PROCESS:
            return

        from rich.table import Table

        table = Table(title="VAE Finetuning Configuration", show_header=True, header_style="bold green")
        table.add_column("Parameter", style="bold white")
        table.add_column("Value", style="bold cyan")

        def flatten(cfg: BaseModel, prefix: str = "") -> list[tuple[str, str]]:
            rows = []
            for field, value in cfg:
                full = f"{prefix}.{field}" if prefix else field
                if isinstance(value, BaseModel):
                    rows.extend(flatten(value, full))
                else:
                    rows.append((full, str(value)[:70]))
            return rows

        for param, value in flatten(config):
            table.add_row(param, value)
        rich.print(table)
