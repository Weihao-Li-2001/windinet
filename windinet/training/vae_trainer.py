# Based on finetune_decoder.py from the WinDiNet development codebase.
# Refactored to use Accelerate for DDP, pydantic configs, and the windinet package.

"""Shockwave CFD VAE finetuning with trainable channel adapters.

The pretrained LTX-Video encoder is frozen. The decoder and the 4->3 / 3->4
adapters are trained using reconstruction, gradient, structural, and wavelet
losses on normalized HDF5 simulation fields.
"""

import math
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import rich
import torch
import torch.distributed as dist
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
from windinet.vae_adapter import (
    AdaptedVAE,
    inflate_vae_io_channels,
    load_adapted_vae,
    load_inflated_vae_checkpoint,
)

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
from windinet.training.vae_visualization import (
    denormalize_fields,
    save_metrics_history,
    save_reconstruction_panels,
)
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
        # Resume bookkeeping. Both the model weights and the training state are
        # read into memory in __init__ (before train() may clean output_dir), so
        # resuming into the same output_dir is safe.
        self._start_epoch = 1
        self._resume_global_step = 0
        self._resume_state: Optional[dict] = None
        self._print_config(config)
        self._setup_accelerator()
        self._load_vae()
        self._collect_trainable_params()
        self._load_resume_state()
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
        self._adapter_meta = None
        self._inflated = False
        if adapter_cfg.enabled and adapter_cfg.mode == "adapter":
            # When resuming, the resume checkpoint supplies the finetuned
            # decoder+adapter weights (same safetensors layout load_adapted_vae reads).
            self._vae, meta = load_adapted_vae(
                self._vae,
                ckpt_path=self._config.resume_from or adapter_cfg.checkpoint,
                device="cpu",
                dtype=torch.float32,
                channels=adapter_cfg.channels,
                k=adapter_cfg.hidden_channels,
                activation=adapter_cfg.activation,
                identity_init=adapter_cfg.identity_init,
                default_temb=adapter_cfg.default_temb,
            )
            logger.info(
                "VAE adapters enabled: "
                f"channels={meta['channels']}, hidden_channels={meta['k']}, "
                f"activation={meta['activation']}"
            )
            self._adapter_meta = meta
        elif adapter_cfg.enabled and adapter_cfg.mode == "inflate":
            n = len(adapter_cfg.channels)
            inflate_vae_io_channels(self._vae, n=n, init=adapter_cfg.inflate_init)
            self._inflated = True
            logger.info(
                f"VAE inflated to {n} native channels "
                f"(init={adapter_cfg.inflate_init}); encoder.conv_in is trainable"
            )
            # Resume: load the finetuned decoder + grown encoder.conv_in weights
            # over the freshly inflated (pretrained-init) VAE.
            if self._config.resume_from is not None:
                meta = load_inflated_vae_checkpoint(
                    self._vae,
                    ckpt_path=self._config.resume_from,
                    device="cpu",
                    dtype=torch.float32,
                )
                logger.info(
                    f"Resumed inflated VAE weights from {self._config.resume_from} "
                    f"(checkpoint epoch={meta.get('epoch', '?')})"
                )

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

        # Inflated mode: the grown input projection must train so the extra
        # channel(s) can actually reach the encoder.
        if self._inflated:
            for p in self._get_encoder_conv_in().parameters():
                p.requires_grad_(True)

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

    def _get_encoder(self) -> nn.Module:
        vae = self._unwrap_vae()
        base_vae = vae.vae if isinstance(vae, AdaptedVAE) else vae
        return base_vae.encoder

    def _get_encoder_conv_in(self) -> nn.Module:
        return self._get_encoder().conv_in

    def _collect_trainable_params(self) -> None:
        self._trainable_params = [p for p in self._vae.parameters() if p.requires_grad]
        vae = self._unwrap_vae()
        base_vae = vae.vae if isinstance(vae, AdaptedVAE) else vae
        # In inflate mode the grown encoder.conv_in is intentionally trainable;
        # everything else in the encoder must still be frozen.
        conv_in_trainable = (
            sum(p.numel() for p in self._get_encoder_conv_in().parameters() if p.requires_grad)
            if self._inflated else 0
        )
        encoder_trainable = sum(p.numel() for p in base_vae.encoder.parameters() if p.requires_grad)
        if encoder_trainable != conv_in_trainable:
            raise RuntimeError(
                f"Only encoder.conv_in may be trainable, but {encoder_trainable:,} encoder "
                f"parameters are trainable ({conv_in_trainable:,} expected)"
            )

        decoder_trainable = sum(p.numel() for p in self._get_decoder().parameters() if p.requires_grad)
        in_adapter_trainable = (
            sum(p.numel() for p in vae.in_adapter.parameters() if p.requires_grad)
            if isinstance(vae, AdaptedVAE) else 0
        )
        out_adapter_trainable = (
            sum(p.numel() for p in vae.out_adapter.parameters() if p.requires_grad)
            if isinstance(vae, AdaptedVAE) else 0
        )
        logger.info(
            "Trainable parameters: "
            f"encoder_conv_in={conv_in_trainable:,}, decoder={decoder_trainable:,}, "
            f"in_adapter={in_adapter_trainable:,}, out_adapter={out_adapter_trainable:,}"
        )

    # ------------------------------------------------------------------
    # Resume (optimizer / scheduler / loss weights / RNG state)
    # ------------------------------------------------------------------

    @staticmethod
    def _state_file_for(ckpt_path: str | Path) -> Path:
        p = Path(ckpt_path)
        # vae_shockwave_epoch005.safetensors -> vae_shockwave_epoch005.state.pt
        return p.parent / (p.stem + ".state.pt")

    def _load_resume_state(self) -> None:
        """Read the sibling training-state file into memory (before any dir wipe).

        The optimizer/scheduler are only rebuilt inside ``train``; here we just
        stash the loaded dict and the epoch/step to resume from.
        """
        if self._config.resume_from is None:
            return
        state_path = self._state_file_for(self._config.resume_from)
        if not state_path.exists():
            logger.warning(
                f"resume_from set but no training-state file at {state_path}; "
                "model weights will be warm-started, but optimizer/scheduler/epoch "
                "will start fresh (epoch 1)."
            )
            return
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        self._resume_state = state
        saved_epoch = int(state.get("epoch", 0))
        self._start_epoch = saved_epoch + 1
        self._resume_global_step = int(state.get("global_opt_step", 0))
        logger.info(
            f"Loaded training state from {state_path}: resuming at epoch "
            f"{self._start_epoch} (saved epoch {saved_epoch}, "
            f"global_opt_step {self._resume_global_step})"
        )

    def _apply_resume_state(self, optimizer, scheduler) -> None:
        """Restore optimizer/scheduler/loss-weighter/RNG from the stashed state."""
        state = self._resume_state
        if state is None:
            return
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        if state.get("loss_weighter") is not None:
            # Loss weighters have no state_dict; restore their mutable attributes
            # (weights, previous/initial losses) without replacing the object.
            self.loss_weighter.__dict__.update(state["loss_weighter"])
        rng = state.get("rng")
        if rng is not None:
            import random as _random

            import numpy as _np

            if rng.get("python") is not None:
                _random.setstate(rng["python"])
            if rng.get("numpy") is not None:
                _np.random.set_state(rng["numpy"])
            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        logger.info("Restored optimizer, scheduler, loss weights, and RNG state.")

    def _set_trainable_modules_mode(self, training: bool) -> None:
        self._get_decoder().train(training)
        vae = self._unwrap_vae()
        if isinstance(vae, AdaptedVAE):
            vae.in_adapter.train(training)
            vae.out_adapter.train(training)
        if self._inflated:
            self._get_encoder_conv_in().train(training)

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

    def _sync_grads(self) -> None:
        """Average trainable-parameter gradients across processes.

        The VAE is deliberately not wrapped in DDP (see ``train``), because the
        training path calls ``vae.encode``/``vae.decode`` directly rather than
        ``forward``. Gradients are therefore all-reduced by hand once per
        optimizer step. All-reduce is linear, so summing the accumulated grads
        and dividing by the world size matches DDP's per-microbatch averaging.
        """
        world_size = self._accelerator.num_processes
        for p in self._trainable_params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(world_size)

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
        # Randomize the split. The HDF5 groups are sorted by the physical
        # parameter gamma, so a contiguous tail slice would put entire gamma
        # regimes (e.g. gamma=1.76) exclusively in eval -- turning validation
        # into an extrapolation test and starving training of those regimes.
        # Shuffle with a fixed seed so both splits span all regimes and the
        # partition stays reproducible across runs.
        split_generator = torch.Generator().manual_seed(cfg.seed)
        perm = torch.randperm(len(full_dataset), generator=split_generator).tolist()
        train_set = Subset(full_dataset, perm[:n_train])
        eval_set = Subset(full_dataset, perm[n_train:n_train + n_eval])

        train_loader = DataLoader(
            train_set,
            batch_size=cfg.optimization.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_dataloader_workers,
            drop_last=True,
        )
        eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False)

        logger.info(f"Dataset: {n_train} train, {n_eval} eval samples")

        # Optimizer. The channel interface (adapters, or the inflated
        # encoder.conv_in) is the 4<->3 bottleneck and benefits from a higher LR,
        # so it goes in its own param group. The LR schedulers below scale each
        # group from its own base LR, preserving the ratio through warmup and
        # cosine annealing.
        base_lr = cfg.optimization.learning_rate
        fast_lr = base_lr * cfg.optimization.adapter_lr_multiplier
        vae = self._unwrap_vae()
        decoder_params = [p for p in self._get_decoder().parameters() if p.requires_grad]
        fast_params = []
        fast_name = "adapter"
        if isinstance(vae, AdaptedVAE):
            fast_params = (
                [p for p in vae.in_adapter.parameters() if p.requires_grad]
                + [p for p in vae.out_adapter.parameters() if p.requires_grad]
            )
        elif self._inflated:
            fast_params = [p for p in self._get_encoder_conv_in().parameters() if p.requires_grad]
            fast_name = "encoder.conv_in"
        param_groups = [{"params": decoder_params, "lr": base_lr}]
        if fast_params:
            param_groups.append({"params": fast_params, "lr": fast_lr})
            logger.info(
                f"{fast_name} LR = {fast_lr:.2e} "
                f"({cfg.optimization.adapter_lr_multiplier:g}x decoder LR {base_lr:.2e})"
            )
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=cfg.optimization.weight_decay,
        )

        # Prepare with Accelerate. The VAE is intentionally NOT wrapped in DDP:
        # the training path calls vae.encode()/decode() directly, which a
        # DistributedDataParallel wrapper does not expose, and DDP would only
        # sync gradients through .forward() anyway. Instead we move it to the
        # device, apply mixed precision via accelerator.autocast(), and
        # all-reduce the trainable gradients manually at each optimizer step.
        self._vae.to(self._accelerator.device)
        optimizer = self._accelerator.prepare(optimizer)
        train_loader = self._accelerator.prepare(train_loader)

        # Size the LR schedule from the per-process (sharded) loader length.
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

        # Restore optimizer/scheduler/loss-weights/RNG when resuming. Done after
        # both are built (load_state_dict needs the target objects) and after
        # set_seed above, so the restored RNG wins. Expose them so end-of-epoch
        # checkpoints can serialize the full training state.
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._apply_resume_state(optimizer, scheduler)

        self._prepare_output_dir()
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

        global_opt_step = self._resume_global_step
        saved_path = None
        metrics_history: list[dict[str, float]] = []

        if self._start_epoch > cfg.optimization.epochs:
            raise ValueError(
                f"resume epoch {self._start_epoch} exceeds configured epochs "
                f"{cfg.optimization.epochs}; increase optimization.epochs to train further."
            )
        if self._start_epoch > 1:
            logger.info(
                f"Resuming VAE finetuning at epoch {self._start_epoch} / "
                f"{cfg.optimization.epochs}"
            )
        else:
            logger.info("Starting VAE decoder finetuning...")

        with live:
            for epoch in range(self._start_epoch, cfg.optimization.epochs + 1):
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
                    with self._accelerator.autocast():
                        recon, _ = self._forward_pass(x)
                    # Match Accelerate's convert_outputs_to_fp32: losses (e.g. the
                    # SSIM conv) run in fp32, so cast the autocast output back.
                    recon = recon.float()
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
                        if self._accelerator.num_processes > 1:
                            self._sync_grads()
                        if cfg.optimization.max_grad_norm > 0:
                            self._accelerator.clip_grad_norm_(self._trainable_params, cfg.optimization.max_grad_norm)

                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_opt_step += 1

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
                    val_metrics = self._evaluate(eval_loader, device)
                    lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"Epoch {epoch}: train_loss={avg_loss:.6f}  "
                        f"val_loss={val_metrics['total_loss']:.6f}  "
                        f"val_VRMSE={val_metrics['vrmse']:.6f}  lr={lr:.2e}"
                    )

                    metrics_row = {
                        "epoch": epoch,
                        "learning_rate": lr,
                        "train_total_loss": avg_loss,
                        "val_total_loss": val_metrics["total_loss"],
                        "val_vrmse": val_metrics["vrmse"],
                        **{f"train_{name}": value for name, value in epoch_losses.items()},
                        **{
                            f"val_{name}": val_metrics[name]
                            for name in ("rmse", "h1", "ssim", "mlw")
                        },
                    }
                    metrics_history.append(metrics_row)
                    csv_path, curve_path = save_metrics_history(metrics_history, cfg.output_dir)
                    logger.info(
                        f"Metrics updated: {csv_path.relative_to(cfg.output_dir)}, "
                        f"{curve_path.relative_to(cfg.output_dir)}"
                    )

                    self._log_metrics({
                        "epoch": epoch,
                        "epoch/train_loss": avg_loss,
                        "epoch/val_loss": val_metrics["total_loss"],
                        "epoch/eval_vrmse": val_metrics["vrmse"],
                        "epoch/learning_rate": lr,
                    })

                    vis_cfg = cfg.visualization
                    if vis_cfg.enabled and epoch % vis_cfg.interval_epochs == 0:
                        self._save_visualization(eval_loader, device, epoch)

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
    def _evaluate(self, eval_loader: DataLoader, device: torch.device) -> dict[str, float]:
        """Evaluate the same reconstruction objective used for training plus VRMSE."""
        self._set_trainable_modules_mode(False)
        sums = {"total_loss": 0.0, "vrmse": 0.0, "rmse": 0.0, "h1": 0.0, "ssim": 0.0, "mlw": 0.0}
        count = 0
        weights = self.loss_weighter.get_weights()
        for batch in eval_loader:
            orig_F = batch["density"].shape[1]
            x = build_shockwave_video(
                batch,
                device=device,
                channel_mean=self._config.data.channel_mean,
                channel_std=self._config.data.channel_std,
                normalization_clip=self._config.data.normalization_clip,
            )
            with self._accelerator.autocast():
                recon, _ = self._forward_pass(x)
            recon = recon.float()
            target = x[:, :, :orig_F]
            recon = recon[:, :, :orig_F]
            losses = reconstruction_losses(
                pred=recon,
                target=target,
                ssim_module=self.ssim_loss,
                wavelet=self._config.loss.wavelet,
                spatial_level=self._config.loss.spatial_level,
                temporal_level=self._config.loss.temporal_level,
                mlw_beta=self._config.loss.mlw_beta,
                mlw_eps=self._config.loss.mlw_eps,
            )
            sums["total_loss"] += float(sum(weights[name] * value for name, value in losses.items()).item())
            sums["vrmse"] += vrmse(recon, target)
            for name, value in losses.items():
                sums[name] += float(value.item())
            count += 1
        self._set_trainable_modules_mode(True)
        return {name: value / max(1, count) for name, value in sums.items()}

    @torch.no_grad()
    def _save_visualization(self, eval_loader: DataLoader, device: torch.device, epoch: int) -> None:
        """Render fixed validation samples at configured physical frame numbers."""
        cfg = self._config
        vis_cfg = cfg.visualization
        self._set_trainable_modules_mode(False)
        saved_count = 0
        for sample_index, batch in enumerate(eval_loader):
            if sample_index >= vis_cfg.num_samples:
                break
            orig_F = batch["density"].shape[1]
            x = build_shockwave_video(
                batch,
                device=device,
                channel_mean=cfg.data.channel_mean,
                channel_std=cfg.data.channel_std,
                normalization_clip=cfg.data.normalization_clip,
            )
            with self._accelerator.autocast():
                recon, _ = self._forward_pass(x)
            recon = recon.float()
            target = denormalize_fields(
                x[:, :, :orig_F],
                cfg.data.channel_mean,
                cfg.data.channel_std,
                cfg.data.normalization_clip,
            )
            prediction = denormalize_fields(
                recon[:, :, :orig_F],
                cfg.data.channel_mean,
                cfg.data.channel_std,
                cfg.data.normalization_clip,
            )
            sample_id_value = batch.get("id", [f"sample_{sample_index:04d}"])
            sample_id = str(sample_id_value[0] if isinstance(sample_id_value, (list, tuple)) else sample_id_value)
            paths = save_reconstruction_panels(
                prediction=prediction[0],
                target=target[0],
                sample_id=sample_id,
                epoch=epoch,
                frame_numbers=vis_cfg.frame_numbers,
                channel_names=cfg.adapter.channels,
                output_dir=cfg.output_dir,
                dpi=vis_cfg.dpi,
            )
            saved_count += len(paths)
        self._set_trainable_modules_mode(True)
        logger.info(f"Saved {saved_count} validation reconstruction PNGs for epoch {epoch}")

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
        if self._inflated:
            # The grown input projection is trained too and must be restored to
            # rebuild an inflated VAE at inference time.
            tensors.update({
                f"encoder_conv_in.{k}": v.detach().cpu().contiguous()
                for k, v in self._get_encoder_conv_in().state_dict().items()
            })

        metadata = {
            "format": "ltx-inflated-io-v1" if self._inflated else "ltx-decoder-plus-adapters-v1",
            "mode": adapter_cfg.mode,
            "inflate_init": adapter_cfg.inflate_init,
            "channels": str(vae.channels if isinstance(vae, AdaptedVAE) else adapter_cfg.channels),
            "n": str(len(vae.channels) if isinstance(vae, AdaptedVAE) else len(adapter_cfg.channels)),
            "k": str(vae.k if isinstance(vae, AdaptedVAE) else 0),
            "activation": str(
                self._adapter_meta["activation"] if self._adapter_meta else adapter_cfg.activation
            ),
            "identity_init": str(
                self._adapter_meta["identity_init"] if self._adapter_meta else adapter_cfg.identity_init
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

        self._save_training_state(path, epoch, step)
        return path

    def _save_training_state(self, ckpt_path: Path, epoch: int, step: int) -> Path:
        """Persist optimizer/scheduler/loss-weights/RNG next to the model checkpoint.

        Written as ``<stem>.state.pt`` alongside the safetensors so that
        ``resume_from=<that safetensors>`` can restore an exact continuation.
        The model weights themselves live in the safetensors (loaded separately),
        so they are intentionally not duplicated here.
        """
        import random

        import numpy as np

        state_path = self._state_file_for(ckpt_path)
        rng = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        # Loss weighters expose no state_dict; snapshot their mutable attributes.
        loss_weighter_state = dict(getattr(self.loss_weighter, "__dict__", {}))
        state = {
            "epoch": epoch,
            "global_opt_step": step,
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "loss_weighter": loss_weighter_state,
            "rng": rng,
        }
        torch.save(state, state_path)
        logger.info(
            f"Training state saved: {state_path.relative_to(self._config.output_dir)}"
        )
        return state_path

    def _prepare_output_dir(self) -> None:
        """Create the output dir, optionally wiping it first for a clean run."""
        out = Path(self._config.output_dir)
        if IS_MAIN_PROCESS:
            if self._config.clean_output_dir and out.exists():
                shutil.rmtree(out)
                logger.info(f"Cleaned output directory: {out}")
            out.mkdir(parents=True, exist_ok=True)
        # Barrier so worker processes never touch the dir before it is ready.
        self._accelerator.wait_for_everyone()

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
