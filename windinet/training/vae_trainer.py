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
from collections import defaultdict
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
    vrms_loss,
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
    """Variance-normalized RMSE, averaged over the batch.

    Thin float-returning wrapper around windinet.losses.vrms_loss (the
    differentiable version used as an optional training loss) -- kept
    separate because this one is @torch.no_grad() and always reports a
    plain float, for the "val_vrmse" eval metric specifically.
    """
    return float(vrms_loss(pred.float(), target.float(), eps=eps).item())


class VaeTrainer:
    def __init__(self, config: VaeTrainerConfig) -> None:
        self._config = config
        # Resume bookkeeping. Both the model weights and the training state are
        # read into memory in __init__ (before train() may clean output_dir), so
        # resuming into the same output_dir is safe.
        self._start_epoch = 1
        self._resume_global_step = 0
        self._resume_state: Optional[dict] = None
        # Best-checkpoint tracking. Paths of the per-epoch checkpoints written so
        # far, oldest first, for keep_last_n pruning.
        self._best_metric_value = math.inf
        self._best_epoch: int | None = None
        self._best_ckpt_path: Path | None = None
        self._checkpoint_paths: list[Path] = []
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
            copy_from_index = (
                adapter_cfg.channels.index(adapter_cfg.inflate_copy_channel)
                if adapter_cfg.inflate_init == "copy"
                else None
            )
            inflate_vae_io_channels(self._vae, n=n, init=adapter_cfg.inflate_init, copy_from_index=copy_from_index)
            self._inflated = True
            copy_suffix = f", copy_from={adapter_cfg.inflate_copy_channel!r}" if adapter_cfg.inflate_init == "copy" else ""
            logger.info(
                f"VAE inflated to {n} native channels "
                f"(init={adapter_cfg.inflate_init}{copy_suffix}); encoder.conv_in is trainable"
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
        # channel(s) can actually reach the encoder -- unless the latent space is
        # deliberately being held fixed, in which case the decoder trains alone
        # and every already-encoded latent stays valid.
        if self._inflated and not adapter_cfg.freeze_conv_in:
            for p in self._get_encoder_conv_in().parameters():
                p.requires_grad_(True)
        elif self._inflated:
            logger.info(
                "encoder.conv_in FROZEN (freeze_conv_in=true): decoder-only refinement. "
                "The latent space is unchanged, so existing latents and any DiT trained on "
                "them remain valid and this VAE can be swapped in at inference."
            )

        # Encoder tail: down_blocks[-1] + mid_block + norm_out + conv_out. These run
        # entirely on the already-4x4 grid (down_blocks[-1] does not spatially
        # downsample), so unfreezing them cannot change the compression ratio -- only
        # what the 512->128 channel projection keeps.
        #
        # Encoder down_blocks[0:3]: the actual spatial-downsampling stages, selected
        # individually via unfreeze_down_blocks so head-vs-tail unfreezing experiments
        # can be composed freely (see windinet/config.py VaeAdapterConfig).
        self._tail_unfrozen = adapter_cfg.unfreeze_encoder_tail
        self._unfrozen_down_blocks = adapter_cfg.unfreeze_down_blocks
        extra_modules = self._get_encoder_extra_modules()
        if extra_modules:
            for module in extra_modules:
                for p in module.parameters():
                    p.requires_grad_(True)
            parts = []
            if self._unfrozen_down_blocks:
                parts.append(f"down_blocks{self._unfrozen_down_blocks}")
            if self._tail_unfrozen:
                parts.append("tail (down_blocks[-1]+mid_block+norm_out+conv_out)")
            logger.info(f"Encoder extra modules UNFROZEN: {' + '.join(parts)}")
        self._encoder_extra_unfrozen = bool(extra_modules)

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

    def _get_encoder_tail_modules(self) -> list[nn.Module]:
        encoder = self._get_encoder()
        return [encoder.down_blocks[-1], encoder.mid_block, encoder.norm_out, encoder.conv_out]

    def _get_encoder_downblock_modules(self) -> list[nn.Module]:
        """down_blocks[0:3] selected by adapter.unfreeze_down_blocks (spatial downsampling stages)."""
        encoder = self._get_encoder()
        return [encoder.down_blocks[i] for i in self._config.adapter.unfreeze_down_blocks]

    def _get_encoder_extra_modules(self) -> list[nn.Module]:
        """Every encoder module beyond conv_in made trainable by unfreeze_down_blocks / unfreeze_encoder_tail."""
        modules = list(self._get_encoder_downblock_modules())
        if self._config.adapter.unfreeze_encoder_tail:
            modules += self._get_encoder_tail_modules()
        return modules

    def _collect_trainable_params(self) -> None:
        self._trainable_params = [p for p in self._vae.parameters() if p.requires_grad]
        vae = self._unwrap_vae()
        base_vae = vae.vae if isinstance(vae, AdaptedVAE) else vae
        # In inflate mode the grown encoder.conv_in is intentionally trainable, and
        # optionally the encoder tail (unfreeze_encoder_tail); everything else in the
        # encoder must still be frozen.
        conv_in_trainable = (
            sum(p.numel() for p in self._get_encoder_conv_in().parameters() if p.requires_grad)
            if self._inflated else 0
        )
        extra_trainable = sum(
            p.numel()
            for module in self._get_encoder_extra_modules()
            for p in module.parameters()
            if p.requires_grad
        )
        encoder_trainable = sum(p.numel() for p in base_vae.encoder.parameters() if p.requires_grad)
        expected_encoder_trainable = conv_in_trainable + extra_trainable
        if encoder_trainable != expected_encoder_trainable:
            raise RuntimeError(
                f"Only encoder.conv_in and the modules selected by unfreeze_down_blocks / "
                f"unfreeze_encoder_tail may be trainable, but {encoder_trainable:,} encoder "
                f"parameters are trainable ({expected_encoder_trainable:,} expected)"
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
            f"encoder_conv_in={conv_in_trainable:,}, encoder_extra={extra_trainable:,}, "
            f"decoder={decoder_trainable:,}, in_adapter={in_adapter_trainable:,}, "
            f"out_adapter={out_adapter_trainable:,}"
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
        if self._config.resume_weights_only:
            logger.info(
                "resume_weights_only=true: loaded the checkpoint's weights but ignoring its "
                "training state. Optimizer, LR schedule and epoch counter start fresh, so the "
                "configured learning_rate/epochs take effect (warm restart)."
            )
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
            if rng.get("xpu") is not None and hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.set_rng_state_all(rng["xpu"])
        logger.info("Restored optimizer, scheduler, loss weights, and RNG state.")

    def _set_trainable_modules_mode(self, training: bool) -> None:
        self._get_decoder().train(training)
        vae = self._unwrap_vae()
        if isinstance(vae, AdaptedVAE):
            vae.in_adapter.train(training)
            vae.out_adapter.train(training)
        # A frozen conv_in stays in eval mode with the rest of the frozen encoder.
        if self._inflated and not self._config.adapter.freeze_conv_in:
            self._get_encoder_conv_in().train(training)
        for module in self._get_encoder_extra_modules():
            module.train(training)

    # ------------------------------------------------------------------
    # VAE encode / decode
    # ------------------------------------------------------------------

    def _encode(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode video to normalised latents. video: [B, C, F, H, W].

        Returns (latents, posterior_mean, posterior_logvar): `latents` is
        the rescaled tensor used for reconstruction (unchanged behavior
        from before this returned a single tensor); `posterior_mean`/
        `posterior_logvar` are the RAW encoder distribution (pre-rescale),
        for the optional KL-divergence loss (windinet.losses.kl_divergence)
        -- see reconstruction_losses' latent_mean/latent_logvar args. Free
        to expose: diffusers' encode() computes logvar internally
        regardless of whether anything downstream reads it.
        """
        out = self._vae.encode(video)
        posterior_mean = out.latent_dist.mean
        posterior_logvar = out.latent_dist.logvar
        norm_mean = self._vae.latents_mean.view(1, -1, 1, 1, 1).to(posterior_mean.device, posterior_mean.dtype)
        norm_std = self._vae.latents_std.view(1, -1, 1, 1, 1).to(posterior_mean.device, posterior_mean.dtype)
        sf = float(getattr(self._vae.config, "scaling_factor", 1.0))
        latents = (posterior_mean - norm_mean) * sf / norm_std
        return latents, posterior_mean, posterior_logvar

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

    def _forward_pass(self, x: torch.Tensor) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """Encode → decode through the VAE.

        Returns (reconstruction, original_frames, posterior_mean,
        posterior_logvar) -- the latter two are _encode's raw distribution,
        passed through for the optional KL loss; callers that don't need it
        (e.g. _save_visualization) just discard them.
        """
        orig_F = x.shape[2]
        latents, posterior_mean, posterior_logvar = self._encode(x)
        recon = self._decode(latents)
        return recon[:, :, :orig_F], orig_F, posterior_mean, posterior_logvar

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
        data_root = Path(cfg.data.data_root)
        full_dataset = ShockWaveDataset(
            data_root,
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
        if cfg.data.overfit_sims is not None:
            # Diagnostic mode: the same handful of sims for both loaders. val is
            # then a memorization score, not a generalization score -- the point
            # is to see how far reconstruction can go when generalization is not
            # in the way. Taken from the head of the same shuffled perm, so these
            # sims span gamma regimes rather than being one contiguous block.
            k = min(cfg.data.overfit_sims, len(full_dataset))
            repeat = cfg.data.overfit_repeat
            # The train subset repeats the same k indices so one epoch holds
            # `repeat` optimizer steps; eval sees each sim once.
            train_set = Subset(full_dataset, perm[:k] * repeat)
            eval_set = Subset(full_dataset, perm[:k])
            n_train = n_eval = k
            logger.warning(
                f"OVERFIT DIAGNOSTIC: training and evaluating on the same {k} sims "
                f"(repeated {repeat}x per epoch). val_vrmse measures memorization, "
                "not generalization."
            )
        else:
            train_set = Subset(full_dataset, perm[:n_train])
            eval_set = Subset(full_dataset, perm[n_train:n_train + n_eval])

        train_loader = DataLoader(
            train_set,
            batch_size=cfg.optimization.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_dataloader_workers,
            drop_last=True,
        )
        # Sharded by hand, not accelerator.prepare() (used for train_loader
        # below): prepare()'s default even_batches padding repeats a few
        # samples so every process gets the same batch count, which would
        # double-count them in _evaluate's sums. A strided split has no
        # padding -- every eval sample lands on exactly one rank, shard
        # sizes differ by at most 1. Previously eval ran entirely on
        # IS_MAIN_PROCESS with num_workers=0 (serial, unsharded, no
        # prefetch) while train_loader was 8-way sharded with prefetch
        # workers -- that mismatch is why eval wall-clock was close to
        # train's despite eval_sims being ~5.7x smaller (see EXPERIMENTS.md
        # "sng_pvc throughput diagnostic").
        world_size = self._accelerator.num_processes
        rank = self._accelerator.process_index
        eval_shard = Subset(eval_set, list(range(rank, len(eval_set), world_size)))
        eval_loader = DataLoader(
            eval_shard,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.data.num_dataloader_workers,
        )

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
        if self._encoder_extra_unfrozen:
            extra_lr = base_lr * cfg.optimization.encoder_tail_lr_multiplier
            extra_params = [
                p
                for module in self._get_encoder_extra_modules()
                for p in module.parameters()
                if p.requires_grad
            ]
            param_groups.append({"params": extra_params, "lr": extra_lr})
            extra_desc = []
            if self._unfrozen_down_blocks:
                extra_desc.append(f"down_blocks{self._unfrozen_down_blocks}")
            if self._tail_unfrozen:
                extra_desc.append("tail")
            logger.info(
                f"encoder extra ({'+'.join(extra_desc)}) LR = {extra_lr:.2e} "
                f"({cfg.optimization.encoder_tail_lr_multiplier:g}x decoder LR {base_lr:.2e})"
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

        # One multiplicative LambdaLR for every param group. LambdaLR scales each
        # group from its own initial_lr, so the adapter/decoder ratio is exact for
        # the whole run. (CosineAnnealingLR's eta_min is an *absolute* floor shared
        # by all groups: with base 5e-5 / adapter 5e-4 / eta_min 1e-6 both groups
        # converge onto 1e-6 and the intended 10x silently becomes 1x by the end.)
        sched_type = cfg.optimization.scheduler_type
        floor = cfg.optimization.min_learning_rate / base_lr
        decay_steps = max(1, total_opt_steps - warmup_steps)
        if sched_type == "wsd":
            stable_steps = int(round(decay_steps * cfg.optimization.stable_fraction))
        elif sched_type == "constant":
            stable_steps = decay_steps
        else:  # cosine: decay immediately after warmup
            stable_steps = 0
        anneal_steps = max(1, decay_steps - stable_steps)

        def lr_factor(step: int) -> float:
            if step < warmup_steps:
                w = cfg.optimization.warmup_start_factor
                return w + (1.0 - w) * (step / max(1, warmup_steps))
            t = step - warmup_steps
            if t < stable_steps:
                return 1.0
            p = min(1.0, (t - stable_steps) / anneal_steps)
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * p))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
        logger.info(
            f"LR schedule '{sched_type}': {total_opt_steps} steps "
            f"({warmup_steps} warmup, {stable_steps} stable, {anneal_steps} anneal), "
            f"peak {base_lr:.2e} -> floor {cfg.optimization.min_learning_rate:.2e}"
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
                epoch_t0 = time.time()
                self._set_trainable_modules_mode(True)
                running_loss = 0.0
                count = 0

                # defaultdict, not a fixed 4-key dict: reconstruction_losses
                # now always includes h2/pcc/vrms too (and "kl" whenever
                # this call site passes latent_mean/latent_logvar), and any
                # loss name absent from a given config's weights just
                # contributes 0 (see the .get(name, 0.0) below) rather than
                # needing every dict here to be kept in sync by hand.
                loss_sum = defaultdict(float)

                grad_norm_sum = defaultdict(float)

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
                        channel_order=cfg.data.channel_order,
                    )
                    with self._accelerator.autocast():
                        recon, _, posterior_mean, posterior_logvar = self._forward_pass(x)
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
                        latent_mean=posterior_mean.float(),
                        latent_logvar=posterior_logvar.float(),
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

                    # .get(name, 0.0): a loss name present in `losses` (every
                    # component reconstruction_losses computes) but absent
                    # from this config's weights contributes 0 rather than
                    # KeyError -- lets the opt-in losses (h2/pcc/vrms/kl)
                    # exist without every existing config having to list
                    # them.
                    total_loss = sum(
                        weights.get(name, 0.0) * value
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
                train_elapsed = time.time() - epoch_t0

                # Every process evaluates its own shard of eval_loader and
                # all-reduces inside _evaluate -- must run unconditionally
                # (not gated on IS_MAIN_PROCESS below), since the all-reduce
                # needs every rank to actually call it or the others hang.
                eval_t0 = time.time()
                val_metrics = self._evaluate(eval_loader, device)
                eval_elapsed = time.time() - eval_t0

                if IS_MAIN_PROCESS:
                    avg_loss = running_loss / max(count, 1)
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
                        # Every val_metrics key except the two already
                        # special-cased above -- dynamically covers h2/pcc/
                        # vrms (always present) and kl (present iff this
                        # run passed latent_mean/latent_logvar), not just
                        # the original four.
                        **{
                            f"val_{name}": value
                            for name, value in val_metrics.items()
                            if name not in ("total_loss", "vrmse")
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

                    vis_t0 = time.time()
                    vis_cfg = cfg.visualization
                    if vis_cfg.enabled and epoch % vis_cfg.interval_epochs == 0:
                        self._save_visualization(eval_loader, device, epoch)
                    vis_elapsed = time.time() - vis_t0

                    monitored = (
                        val_metrics["vrmse"]
                        if cfg.checkpoints.best_metric == "val_vrmse"
                        else val_metrics["total_loss"]
                    )
                    improved = monitored < self._best_metric_value
                    if improved:
                        self._best_metric_value = monitored
                        self._best_epoch = epoch

                    ckpt_t0 = time.time()
                    if cfg.checkpoints.interval and epoch % cfg.checkpoints.interval == 0:
                        saved_path = self._save_checkpoint(
                            epoch, global_opt_step, improved=improved
                        )
                    ckpt_elapsed = time.time() - ckpt_t0

                    logger.info(
                        f"Epoch {epoch} timing: train={train_elapsed:.1f}s "
                        f"eval={eval_elapsed:.1f}s viz={vis_elapsed:.1f}s "
                        f"ckpt={ckpt_elapsed:.1f}s total={time.time() - epoch_t0:.1f}s"
                    )

                self._accelerator.wait_for_everyone()

        # Final checkpoint. improved=False so a last epoch that is worse than the
        # best cannot overwrite the best weights; it still refreshes the rolling
        # last/ pair. In per-epoch mode this is the usual unconditional save.
        if IS_MAIN_PROCESS:
            saved_path = self._save_checkpoint(
                cfg.optimization.epochs,
                global_opt_step,
                improved=not cfg.checkpoints.save_best_only,
            )
            if cfg.checkpoints.save_best_only and self._best_epoch is not None:
                logger.info(
                    f"Best epoch: {self._best_epoch} "
                    f"({cfg.checkpoints.best_metric}={self._best_metric_value:.6f})"
                )

        if self._wandb_run is not None:
            self._wandb_run.finish()

        self._accelerator.end_training()
        return saved_path

    @torch.no_grad()
    def _evaluate(self, eval_loader: DataLoader, device: torch.device) -> dict[str, float]:
        """Evaluate the same reconstruction objective used for training plus VRMSE.

        Called on every process, each over its own shard of ``eval_loader``
        (see the strided split in ``train``). Sums are all-reduced by hand
        before averaging so every process returns the identical, exact,
        whole-eval-set average -- same manual-all-reduce-instead-of-DDP
        pattern as ``_sync_grads``, for the same reason (the VAE isn't
        DDP-wrapped).
        """
        self._set_trainable_modules_mode(False)
        # defaultdict: reconstruction_losses' opt-in components (h2/pcc/vrms/
        # kl) don't need a hardcoded slot here, same reasoning as loss_sum
        # in train().
        sums = defaultdict(float)
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
                channel_order=self._config.data.channel_order,
            )
            with self._accelerator.autocast():
                recon, _, posterior_mean, posterior_logvar = self._forward_pass(x)
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
                latent_mean=posterior_mean.float(),
                latent_logvar=posterior_logvar.float(),
            )
            # .get(name, 0.0): see the matching comment in train()'s backward
            # total_loss -- an opt-in loss absent from this config's weights
            # contributes 0 rather than KeyError.
            sums["total_loss"] += float(sum(weights.get(name, 0.0) * value for name, value in losses.items()).item())
            sums["vrmse"] += vrmse(recon, target)
            for name, value in losses.items():
                sums[name] += float(value.item())
            count += 1
        self._set_trainable_modules_mode(True)

        if self._accelerator.num_processes > 1:
            keys = list(sums.keys())
            # float32, matching every other tensor dtype in this trainer (see
            # load_vae(..., dtype=torch.float32) etc.) -- fp64 collectives
            # aren't reliably supported on the XPU/oneCCL backend this runs
            # on, and float32 has ample precision for summing ~100 loss
            # values per rank.
            totals = torch.tensor([sums[k] for k in keys] + [float(count)], device=device, dtype=torch.float32)
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            *summed, total_count = totals.tolist()
            sums = dict(zip(keys, summed))
            count = total_count

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
                channel_order=cfg.data.channel_order,
            )
            with self._accelerator.autocast():
                recon, _, _, _ = self._forward_pass(x)
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

    def _save_checkpoint(self, epoch: int, step: int, *, improved: bool = True) -> Path:
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
        if self._unfrozen_down_blocks:
            # NOTE: load_inflated_vae_checkpoint() (windinet/vae_adapter.py) does not
            # yet restore these keys -- resuming a down-block-unfrozen run isn't wired
            # up. Fine for a diagnostic (checkpoints disabled, resume_from: null);
            # revisit before using unfreeze_down_blocks on a real training run.
            for idx, module in zip(self._unfrozen_down_blocks, self._get_encoder_downblock_modules()):
                tensors.update({
                    f"encoder_down_block_{idx}.{k}": v.detach().cpu().contiguous()
                    for k, v in module.state_dict().items()
                })
        if self._tail_unfrozen:
            # NOTE: load_inflated_vae_checkpoint() (windinet/vae_adapter.py) does not
            # yet restore these keys -- resuming a tail-unfrozen run isn't wired up.
            # Fine for a diagnostic (checkpoints disabled, resume_from: null); revisit
            # before using unfreeze_encoder_tail on a real training run.
            for name, module in zip(
                ("down_blocks_last", "mid_block", "norm_out", "conv_out"),
                self._get_encoder_tail_modules(),
            ):
                tensors.update({
                    f"encoder_tail.{name}.{k}": v.detach().cpu().contiguous()
                    for k, v in module.state_dict().items()
                })

        metadata = {
            "format": "ltx-inflated-io-v1" if self._inflated else "ltx-decoder-plus-adapters-v1",
            "mode": adapter_cfg.mode,
            "inflate_init": adapter_cfg.inflate_init,
            "inflate_copy_channel": str(adapter_cfg.inflate_copy_channel),
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

        ckpt_cfg = self._config.checkpoints
        if not ckpt_cfg.save_best_only:
            path = save_dir / f"vae_shockwave_epoch{epoch:03d}.safetensors"
            save_file(tensors, path, metadata=metadata)
            logger.info(f"VAE checkpoint saved: {path.relative_to(self._config.output_dir)}")
            self._save_training_state(self._state_file_for(path), epoch, step)
            # train() saves a final checkpoint for the last epoch on top of that
            # epoch's own save, so the same path can arrive here twice.
            if path not in self._checkpoint_paths:
                self._checkpoint_paths.append(path)
            self._prune_checkpoints()
            return path

        # save_best_only: two fixed slots instead of one pair per epoch.
        #   best/ -- weights only, replaced when the monitored metric improves.
        #   last/ -- weights + optimizer state, replaced every time, so a resume
        #            always pairs weights with the moments that produced them.
        if improved:
            best_path = save_dir / "vae_shockwave_best.safetensors"
            save_file(tensors, best_path, metadata={**metadata, "best_metric": ckpt_cfg.best_metric})
            self._best_ckpt_path = best_path
            logger.info(
                f"New best checkpoint (epoch {epoch}, {ckpt_cfg.best_metric}="
                f"{self._best_metric_value:.6f}): {best_path.relative_to(self._config.output_dir)}"
            )

        if ckpt_cfg.save_last_state:
            last_path = save_dir / "vae_shockwave_last.safetensors"
            save_file(tensors, last_path, metadata=metadata)
            self._save_training_state(self._state_file_for(last_path), epoch, step)

        # Return the best weights when we have them: that is what downstream
        # (inference, DiT latent re-encoding) should consume.
        return self._best_ckpt_path or (save_dir / "vae_shockwave_last.safetensors")

    def _prune_checkpoints(self) -> None:
        """Delete all but the newest keep_last_n per-epoch checkpoints."""
        keep = self._config.checkpoints.keep_last_n
        if not (0 < keep < len(self._checkpoint_paths)):
            return
        for stale in self._checkpoint_paths[:-keep]:
            for f in (stale, self._state_file_for(stale)):
                try:
                    f.unlink(missing_ok=True)
                except OSError as e:  # a missing file is fine; anything else is worth seeing
                    logger.warning(f"Could not remove stale checkpoint {f}: {e}")
            logger.info(f"Pruned old checkpoint: {stale.name} (keep_last_n={keep})")
        self._checkpoint_paths = self._checkpoint_paths[-keep:]

    def _save_training_state(self, state_path: Path, epoch: int, step: int) -> Path:
        """Persist optimizer/scheduler/loss-weights/RNG next to the model checkpoint.

        Written as ``<stem>.state.pt`` alongside the safetensors so that
        ``resume_from=<that safetensors>`` can restore an exact continuation.
        The model weights themselves live in the safetensors (loaded separately),
        so they are intentionally not duplicated here.
        """
        import random

        import numpy as np

        rng = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "xpu": torch.xpu.get_rng_state_all()
            if hasattr(torch, "xpu") and torch.xpu.is_available()
            else None,
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
            # Default rng_types=["generator"] makes every accelerator.prepare()
            # call broadcast the train_loader's shuffle-sampler RNG state from
            # rank 0 to every other rank over the distributed backend. On
            # sng_pvc's 4-rank ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE launch (job
            # 520303) that broadcast corrupts in transit through oneCCL and
            # crashes with "RuntimeError: Invalid mt19937 state" in
            # accelerate/utils/random.py's synchronize_rng_state -- the
            # identical 8-tile FLAT launches (520300-520302) hit the same code
            # path every epoch without issue, so this is isolated to
            # COMPOSITE, not a general bug in our seeding.
            #
            # It's safe to disable: train() calls accelerate's set_seed(cfg.seed)
            # identically on every rank *before* the DataLoader/sampler is
            # constructed, and no rank-divergent randomness happens between
            # that call and accelerator.prepare(train_loader) (the train/eval
            # split's own torch.Generator is seeded from cfg.seed directly, not
            # the global RNG). So every rank's default RandomSampler already
            # gets an identical generator seed without any cross-rank
            # broadcast; rng_types=[] just skips the (buggy, redundant) sync.
            rng_types=[],
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
