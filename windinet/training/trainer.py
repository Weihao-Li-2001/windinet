# Based on LTX-Video-Trainer by Lightricks (Apache 2.0).
# https://github.com/Lightricks/LTX-Video-Trainer
# Modified: scalar-only conditioning, removed LoRA/text, absorbed training strategy.

import json
import os
import random
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import MagicMock

import rich
import torch
import wandb
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from pydantic import BaseModel, ConfigDict, computed_field
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
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    LRScheduler,
    PolynomialLR,
    StepLR,
)
from torch.utils.data import DataLoader

from windinet.utils import logger, convert_checkpoint, get_gpu_memory_gb
from windinet.config import LtxvTrainerConfig
from windinet.training.datasets import PrecomputedDataset
from windinet.inference.model_loader import load_ltxv_components
from windinet.scalar_embeddings import ScalarEmbedding
from windinet.training.timestep_samplers import SAMPLERS
from windinet.ltxv_utils import get_rope_scale_factors

os.environ["TOKENIZERS_PARALLELISM"] = "true"

IS_MAIN_PROCESS = os.environ.get("LOCAL_RANK", "0") == "0"
if not IS_MAIN_PROCESS:
    from transformers.utils.logging import disable_progress_bar
    disable_progress_bar()

StepCallback = Callable[[int, int, list[Path]], None]

COMPILE_WARMUP_STEPS = 5
MEMORY_CHECK_INTERVAL = 200
DEFAULT_FPS = 24


# ---------------------------------------------------------------------------
# Training batch container (from training_strategies.py)
# ---------------------------------------------------------------------------

class TrainingBatch(BaseModel):
    """Container for prepared training data."""

    latents: Tensor
    targets: Tensor
    prompt_embeds: Tensor
    prompt_attention_mask: Tensor
    scalars: Tensor | None = None
    timesteps: Tensor
    sigmas: Tensor
    conditioning_mask: Tensor
    num_frames: int
    height: int
    width: int
    fps: float
    rope_interpolation_scale: list[float]

    @computed_field
    @property
    def batch_size(self) -> int:
        return self.latents.shape[0]

    @computed_field
    @property
    def sequence_length(self) -> int:
        return self.latents.shape[1]

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# Training statistics
# ---------------------------------------------------------------------------

class TrainingStats(BaseModel):
    total_time_seconds: float
    compilation_time_seconds: Optional[float]
    training_time: float
    steps_per_second: float
    samples_per_second: float
    peak_gpu_memory_gb: float
    global_batch_size: int
    num_processes: int


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class LtxvTrainer:
    def __init__(self, trainer_config: LtxvTrainerConfig) -> None:
        self._config = trainer_config
        self._print_config(trainer_config)
        self._setup_accelerator()
        self._load_models()
        self._init_scalar_embedding()
        self._compile_transformer()
        self._collect_trainable_params()
        self._load_checkpoint()
        self._prepare_models_for_training()
        self._dataset = None
        self._global_step = -1
        self._checkpoint_paths = []
        self._epoch_losses = []
        self._epoch_train_losses = []
        self._init_wandb()

    # ------------------------------------------------------------------
    # Batch preparation (absorbed from training_strategies.py)
    # ------------------------------------------------------------------

    def _get_data_sources(self) -> list[str]:
        """Data sources required for training."""
        sources = ["latents"]
        if self._config.scalar_conditioning.enabled:
            sources.append("scalars")
        return sources

    def _prepare_batch(self, batch: dict[str, Any], timestep_sampler) -> TrainingBatch:
        """Prepare a training batch with noise and conditioning."""
        latents = batch["latents"]
        target_latents = latents["latents"]

        latent_frames = latents["num_frames"][0].item()
        latent_height = latents["height"][0].item()
        latent_width = latents["width"][0].item()

        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values in batch. Found: {fps.tolist()}, using first: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Scalar-only mode: always use zero placeholder text embeddings
        batch_size = target_latents.shape[0]
        prompt_embeds = torch.zeros(batch_size, 1, 4096, device=target_latents.device, dtype=target_latents.dtype)
        prompt_attention_mask = torch.ones(batch_size, 1, device=target_latents.device, dtype=torch.bool)

        # Get scalar conditioning if enabled
        scalars = None
        if self._config.scalar_conditioning.enabled and "scalars" in batch:
            scalars = batch["scalars"]["scalars"]

        # Create conditioning mask (first frame conditioning)
        conditioning_mask = self._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=target_latents.shape[1],
            height=latent_height,
            width=latent_width,
            device=target_latents.device,
        )

        # Create noise for the target latents
        sigmas = timestep_sampler.sample_for(target_latents)
        noise = torch.randn_like(target_latents, device=target_latents.device)

        sigmas = sigmas.view(-1, 1, 1)
        noisy_latents = (1 - sigmas) * target_latents + sigmas * noise

        # For conditioning tokens, use clean latents
        conditioning_mask_expanded = conditioning_mask.unsqueeze(-1)
        noisy_latents = torch.where(conditioning_mask_expanded, target_latents, noisy_latents)

        targets = noise - target_latents

        # Create timesteps based on conditioning mask
        sampled_timestep_values = torch.round(sigmas.squeeze(-1).squeeze(-1) * 1000.0).long()
        timesteps = self._create_timesteps_from_conditioning_mask(conditioning_mask, sampled_timestep_values)

        rope_interpolation_scale_factors = get_rope_scale_factors(fps)

        return TrainingBatch(
            latents=noisy_latents,
            targets=targets,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            scalars=scalars,
            timesteps=timesteps,
            sigmas=sigmas,
            conditioning_mask=conditioning_mask,
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            fps=fps,
            rope_interpolation_scale=rope_interpolation_scale_factors,
        )

    @staticmethod
    def _prepare_model_inputs(batch: TrainingBatch) -> dict[str, Any]:
        """Prepare inputs for the transformer model."""
        return {
            "hidden_states": batch.latents,
            "encoder_hidden_states": batch.prompt_embeds,
            "timestep": batch.timesteps,
            "encoder_attention_mask": batch.prompt_attention_mask,
            "num_frames": batch.num_frames,
            "height": batch.height,
            "width": batch.width,
            "rope_interpolation_scale": batch.rope_interpolation_scale,
            "return_dict": False,
        }

    @staticmethod
    def _compute_loss(model_pred: Tensor, batch: TrainingBatch) -> Tensor:
        """Compute masked MSE loss using conditioning mask."""
        loss = (model_pred - batch.targets).pow(2)
        loss_mask = (~batch.conditioning_mask.unsqueeze(-1)).float()
        loss = loss.mul(loss_mask).div(loss_mask.mean())
        return loss.mean()

    @staticmethod
    def _create_timesteps_from_conditioning_mask(
        conditioning_mask: Tensor, sampled_timestep_values: Tensor
    ) -> Tensor:
        expanded_timesteps = sampled_timestep_values.unsqueeze(1).expand_as(conditioning_mask)
        return torch.where(conditioning_mask, 0, expanded_timesteps)

    def _create_first_frame_conditioning_mask(
        self, batch_size: int, sequence_length: int, height: int, width: int, device: torch.device
    ) -> Tensor:
        conditioning_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=device)
        if (
            self._config.conditioning.first_frame_conditioning_p > 0
            and random.random() < self._config.conditioning.first_frame_conditioning_p
        ):
            first_frame_end_idx = height * width
            if first_frame_end_idx < sequence_length:
                conditioning_mask[:, :first_frame_end_idx] = True
        return conditioning_mask

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        disable_progress_bars: bool = False,
        step_callback: StepCallback | None = None,
    ) -> tuple[Path, TrainingStats]:
        """Start the training process."""
        device = self._accelerator.device
        cfg = self._config
        start_mem = get_gpu_memory_gb(device)

        train_start_time = time.time()
        set_seed(cfg.seed)
        logger.debug(f"Process {self._accelerator.process_index} using seed: {cfg.seed}")

        self._init_optimizer()
        self._init_dataloader()
        data_iter = iter(self._dataloader)
        self._init_timestep_sampler()

        self._accelerator.wait_for_everyone()
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self._save_config()

        logger.info("Starting training...")

        if disable_progress_bars or not IS_MAIN_PROCESS:
            train_progress = MagicMock()
            live = nullcontext()
            if IS_MAIN_PROCESS:
                logger.warning("Progress bars disabled.")
        else:
            train_progress = Progress(
                TextColumn("Training Step"),
                MofNCompleteColumn(),
                BarColumn(bar_width=40, style="blue"),
                TextColumn("Loss: {task.fields[loss]:.4f}"),
                TextColumn("LR: {task.fields[lr]:.2e}"),
                TextColumn("Time/Step: {task.fields[step_time]:.2f}s"),
                TimeElapsedColumn(),
                TextColumn("ETA:"),
                TimeRemainingColumn(compact=True),
            )
            live = Live(Panel(train_progress), refresh_per_second=2)

        self._transformer.train()
        self._global_step = 0

        compilation_time = None
        peak_mem_during_training = start_mem
        actual_training_start = None

        with live:
            task = train_progress.add_task(
                "Training",
                total=cfg.optimization.steps,
                loss=0.0,
                lr=cfg.optimization.learning_rate,
                step_time=0.0,
            )

            for step in range(cfg.optimization.steps * cfg.optimization.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self._dataloader)
                    batch = next(data_iter)

                if step == COMPILE_WARMUP_STEPS and cfg.acceleration.compile_with_inductor:
                    compilation_time = time.time() - train_start_time
                    actual_training_start = time.time()
                elif step == COMPILE_WARMUP_STEPS and not cfg.acceleration.compile_with_inductor:
                    actual_training_start = train_start_time

                step_start_time = time.time()
                with self._accelerator.accumulate(self._transformer):
                    loss = self._training_step(batch)
                    self._accelerator.backward(loss)

                    is_optimization_step = self._accelerator.sync_gradients

                    if is_optimization_step:
                        self._global_step += 1
                        avg_loss = self._accelerator.gather(loss.detach()).mean()

                        if IS_MAIN_PROCESS and (self._global_step % int(os.getenv("LTXV_TRAIN_LOG_EVERY", "1")) == 0):
                            current_lr = self._optimizer.param_groups[0]["lr"]
                            self._accelerator.print(
                                f"[train] step={self._global_step} loss={avg_loss.item():.6f} lr={current_lr:.3e}"
                            )

                        if cfg.optimization.max_grad_norm > 0:
                            self._accelerator.clip_grad_norm_(
                                self._trainable_params,
                                cfg.optimization.max_grad_norm,
                            )

                        self._optimizer.step()
                        self._optimizer.zero_grad()

                        if self._lr_scheduler is not None:
                            self._lr_scheduler.step()

                    if (
                        cfg.checkpoints.interval
                        and is_optimization_step
                        and self._global_step > 0
                        and self._global_step % cfg.checkpoints.interval == 0
                        and IS_MAIN_PROCESS
                    ):
                        self._save_checkpoint()

                    if (
                        cfg.checkpoints.interval
                        and is_optimization_step
                        and self._global_step > 0
                        and self._global_step % cfg.checkpoints.interval == 0
                    ):
                        epoch = self._global_step // cfg.checkpoints.interval
                        if self._epoch_train_losses:
                            epoch_train_loss = sum(self._epoch_train_losses) / len(self._epoch_train_losses)
                            self._epoch_train_losses.clear()
                        else:
                            epoch_train_loss = None

                        if IS_MAIN_PROCESS:
                            train_str = f"{epoch_train_loss:.6f}" if epoch_train_loss else "N/A"
                            self._accelerator.print(
                                f"[epoch {epoch}] step={self._global_step} train_loss={train_str}"
                            )
                            epoch_metrics = {"epoch": epoch}
                            if epoch_train_loss is not None:
                                epoch_metrics["epoch/train_loss"] = epoch_train_loss
                            self._log_metrics(epoch_metrics)

                            self._epoch_losses.append({
                                "epoch": epoch,
                                "step": self._global_step,
                                "train_loss": epoch_train_loss,
                            })
                            self._save_epoch_log()

                    self._accelerator.wait_for_everyone()

                    if step_callback and is_optimization_step:
                        step_callback(self._global_step, cfg.optimization.steps, [])

                    self._accelerator.wait_for_everyone()

                    if IS_MAIN_PROCESS:
                        current_lr = self._optimizer.param_groups[0]["lr"]
                        elapsed = time.time() - train_start_time
                        progress_percentage = self._global_step / cfg.optimization.steps
                        if progress_percentage > 0:
                            total_estimated = elapsed / progress_percentage
                            total_time = f"{total_estimated // 3600:.0f}h {(total_estimated % 3600) // 60:.0f}m"
                        else:
                            total_time = "calculating..."

                        step_time = (time.time() - step_start_time) * cfg.optimization.gradient_accumulation_steps
                        display_loss = avg_loss.item() if is_optimization_step else loss.item()

                        train_progress.update(
                            task,
                            advance=1 if is_optimization_step else 0,
                            loss=display_loss,
                            lr=current_lr,
                            step_time=step_time,
                            total_time=total_time,
                        )

                        if is_optimization_step:
                            self._log_metrics({
                                "train/loss": display_loss,
                                "train/learning_rate": current_lr,
                                "train/step_time": step_time,
                                "train/global_step": self._global_step,
                            })
                            self._epoch_train_losses.append(display_loss)

                        if disable_progress_bars and is_optimization_step and self._global_step % 20 == 0:
                            logger.info(
                                f"Step {self._global_step}/{cfg.optimization.steps} - "
                                f"Loss: {display_loss:.4f}, LR: {current_lr:.2e}, "
                                f"Time/Step: {step_time:.2f}s, Total Time: {total_time}",
                            )

                    if step % MEMORY_CHECK_INTERVAL == 0:
                        current_mem = get_gpu_memory_gb(device)
                        peak_mem_during_training = max(peak_mem_during_training, current_mem)

        train_end_time = time.time()
        end_mem = get_gpu_memory_gb(device)
        peak_mem = max(start_mem, end_mem, peak_mem_during_training)

        if cfg.acceleration.compile_with_inductor:
            training_time = train_end_time - actual_training_start
            steps_per_second = (cfg.optimization.steps - COMPILE_WARMUP_STEPS) / training_time
        else:
            training_time = train_end_time - train_start_time
            steps_per_second = cfg.optimization.steps / training_time

        effective_batch_size = (
            cfg.optimization.batch_size
            * self._accelerator.num_processes
            * cfg.optimization.gradient_accumulation_steps
        )
        samples_per_second = steps_per_second * effective_batch_size

        stats = TrainingStats(
            total_time_seconds=train_end_time - train_start_time,
            training_time=training_time,
            compilation_time_seconds=compilation_time,
            steps_per_second=steps_per_second,
            samples_per_second=samples_per_second,
            peak_gpu_memory_gb=peak_mem,
            num_processes=self._accelerator.num_processes,
            global_batch_size=effective_batch_size,
        )

        final_step = self._global_step if self._global_step > 0 else cfg.optimization.steps
        comfy_path: Path = Path(cfg.output_dir) / "checkpoints" / f"model_weights_step_{final_step:05d}.safetensors"

        train_progress.remove_task(task)
        self._accelerator.end_training()

        if IS_MAIN_PROCESS:
            saved_path = self._save_checkpoint()
            comfy_path = saved_path

            if os.getenv("LTXV_SAVE_COMFY", "0").lower() in ("1", "true", "yes", "y"):
                maybe_comfy = saved_path.parent / f"comfy_{saved_path.name}"
                try:
                    convert_checkpoint(
                        input_path=str(saved_path),
                        to_comfy=True,
                        output_path=str(maybe_comfy),
                    )
                    comfy_path = maybe_comfy
                except Exception as e:
                    logger.warning(f"Comfy conversion failed, continuing without it: {e}")

            self._log_training_stats(stats)

            if self._wandb_run is not None:
                self._log_metrics({
                    "stats/total_time_minutes": stats.total_time_seconds / 60,
                    "stats/training_time_minutes": stats.training_time / 60,
                    "stats/compilation_time_seconds": stats.compilation_time_seconds,
                    "stats/steps_per_second": stats.steps_per_second,
                    "stats/samples_per_second": stats.samples_per_second,
                    "stats/peak_gpu_memory_gb": stats.peak_gpu_memory_gb,
                })
                self._wandb_run.finish()

        self._accelerator.end_training()
        return comfy_path, stats

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def _training_step(self, batch: dict[str, dict[str, Tensor]]) -> Tensor:
        training_batch = self._prepare_batch(batch, self._timestep_sampler)

        if self._scalar_embedding is not None and training_batch.scalars is not None:
            training_batch = self._embed_and_concat_scalars(training_batch)

        model_inputs = self._prepare_model_inputs(training_batch)

        for key, val in model_inputs.items():
            if isinstance(val, torch.Tensor) and val.is_floating_point():
                if torch.isnan(val).any() or torch.isinf(val).any():
                    logger.error(f"NaN/Inf in model_inputs['{key}']: shape={val.shape}")
                    raise ValueError(f"NaN/Inf detected in model_inputs['{key}']")

        model_pred = self._transformer(**model_inputs)[0]

        if torch.isnan(model_pred).any() or torch.isinf(model_pred).any():
            logger.error("NaN/Inf in model_pred after transformer forward pass")
            raise ValueError("NaN/Inf detected in model prediction")

        loss = self._compute_loss(model_pred, training_batch)

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.error(f"NaN/Inf in loss: {loss.item()}")
            raise ValueError("NaN/Inf detected in loss")

        return loss

    def _embed_and_concat_scalars(self, batch: TrainingBatch) -> TrainingBatch:
        """Embed scalar values and use as prompt embeddings."""
        if torch.isnan(batch.scalars).any() or torch.isinf(batch.scalars).any():
            raise ValueError("Input scalars contain NaN or Inf values")

        for name, param in self._scalar_embedding.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                raise ValueError(f"Scalar embedding weights contain NaN or Inf: {name}")

        scalar_embeds = self._scalar_embedding(batch.scalars)

        if torch.isnan(scalar_embeds).any() or torch.isinf(scalar_embeds).any():
            raise ValueError("Scalar embeddings contain NaN or Inf values")

        batch_size = batch.scalars.shape[0]
        num_scalar_tokens = scalar_embeds.shape[1]

        # Scalar-only mode: use only scalar embeddings
        new_prompt_embeds = scalar_embeds
        new_attention_mask = torch.ones(
            batch_size, num_scalar_tokens,
            dtype=batch.prompt_attention_mask.dtype,
            device=batch.prompt_attention_mask.device,
        )

        return batch.model_copy(
            update={
                "prompt_embeds": new_prompt_embeds,
                "prompt_attention_mask": new_attention_mask,
            }
        )

    # ------------------------------------------------------------------
    # Model loading and setup
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        components = load_ltxv_components(
            model_source=self._config.model.model_source,
            transformer_dtype=torch.float32,
            vae_dtype=torch.bfloat16,
        )

        self._scheduler = components.scheduler
        self._vae = components.vae
        self._transformer = components.transformer

        self._vae.requires_grad_(False)
        self._transformer.requires_grad_(False)

    def _init_scalar_embedding(self) -> None:
        if self._config.scalar_conditioning.enabled:
            logger.info(
                f"Initializing scalar embeddings for: {self._config.scalar_conditioning.scalar_names}"
            )
            self._scalar_embedding = ScalarEmbedding(self._config.scalar_conditioning)
        else:
            self._scalar_embedding = None

    def _compile_transformer(self) -> None:
        if not self._config.acceleration.compile_with_inductor:
            return
        torch._dynamo.config.inline_inbuilt_nn_modules = True
        torch._dynamo.config.cache_size_limit = 128
        compile_module = partial(torch.compile, mode=self._config.acceleration.compilation_mode)
        self._transformer.transformer_blocks = nn.ModuleList(
            [compile_module(block) for block in self._transformer.transformer_blocks],
        )

    def _collect_trainable_params(self) -> None:
        self._transformer.requires_grad_(True)
        self._trainable_params = [p for p in self._transformer.parameters() if p.requires_grad]

        if self._scalar_embedding is not None:
            self._trainable_params.extend(self._scalar_embedding.parameters())
            logger.debug(
                f"Scalar embedding params: {sum(p.numel() for p in self._scalar_embedding.parameters()):,}"
            )

        logger.debug(f"Total trainable params count: {sum(p.numel() for p in self._trainable_params):,}")

    def _init_timestep_sampler(self) -> None:
        sampler_cls = SAMPLERS[self._config.flow_matching.timestep_sampling_mode]
        self._timestep_sampler = sampler_cls(**self._config.flow_matching.timestep_sampling_params)

    def _load_checkpoint(self) -> None:
        if not self._config.model.load_checkpoint:
            return

        checkpoint_path = self._find_checkpoint(self._config.model.load_checkpoint)
        if not checkpoint_path:
            logger.warning(f"Could not find checkpoint at {self._config.model.load_checkpoint}")
            return

        transformer = self._accelerator.unwrap_model(self._transformer)

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        state_dict = load_file(checkpoint_path)
        transformer.load_state_dict(state_dict)

        if self._scalar_embedding is not None:
            scalar_checkpoint = self._find_scalar_checkpoint(checkpoint_path)
            if scalar_checkpoint:
                logger.info(f"Loading scalar embedding from {scalar_checkpoint}")
                scalar_state_dict = load_file(scalar_checkpoint)
                self._scalar_embedding.load_state_dict(scalar_state_dict)

    def _find_scalar_checkpoint(self, model_checkpoint: Path) -> Path | None:
        stem = model_checkpoint.stem
        if "_step_" not in stem:
            return None
        step_str = stem.split("_step_")[-1]
        scalar_filename = f"scalar_embedding_step_{step_str}.safetensors"
        scalar_path = model_checkpoint.parent / scalar_filename
        return scalar_path if scalar_path.exists() else None

    def _prepare_models_for_training(self) -> None:
        prepare = self._accelerator.prepare
        self._vae = prepare(self._vae).to("cpu")
        self._transformer = prepare(self._transformer)

        if self._scalar_embedding is not None:
            self._scalar_embedding = prepare(self._scalar_embedding)

        if self._config.optimization.enable_gradient_checkpointing:
            self._transformer.enable_gradient_checkpointing()

    @staticmethod
    def _find_checkpoint(checkpoint_path: str | Path) -> Path | None:
        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_file():
            if not checkpoint_path.suffix == ".safetensors":
                raise ValueError(f"Checkpoint file must have a .safetensors extension: {checkpoint_path}")
            return checkpoint_path

        if checkpoint_path.is_dir():
            checkpoints = list(checkpoint_path.rglob("*step_*.safetensors"))
            if not checkpoints:
                return None

            def _get_step_num(p: Path) -> int:
                try:
                    return int(p.stem.split("step_")[1])
                except (IndexError, ValueError):
                    return -1

            return max(checkpoints, key=_get_step_num)

        raise ValueError(f"Invalid checkpoint path: {checkpoint_path}. Must be a file or directory.")

    # ------------------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------------------

    def _init_optimizer(self) -> None:
        opt_cfg = self._config.optimization
        lr = opt_cfg.learning_rate

        if opt_cfg.optimizer_type == "adamw":
            optimizer = AdamW(self._trainable_params, lr=lr)
        elif opt_cfg.optimizer_type == "adamw8bit":
            from bitsandbytes.optim import AdamW8bit
            optimizer = AdamW8bit(self._trainable_params, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_cfg.optimizer_type}")

        optimizer = self._accelerator.prepare(optimizer)
        lr_scheduler = self._create_scheduler(optimizer)
        self._optimizer, self._lr_scheduler = optimizer, lr_scheduler

    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> LRScheduler | None:
        scheduler_type = self._config.optimization.scheduler_type
        steps = self._config.optimization.steps
        params = dict(self._config.optimization.scheduler_params or {})

        if scheduler_type is None or scheduler_type == "constant":
            return None

        if scheduler_type == "linear":
            return LinearLR(optimizer, start_factor=params.pop("start_factor", 1.0),
                            end_factor=params.pop("end_factor", 0.1), total_iters=steps, **params)
        elif scheduler_type == "cosine":
            return CosineAnnealingLR(optimizer, T_max=steps, eta_min=params.pop("eta_min", 0), **params)
        elif scheduler_type == "cosine_with_restarts":
            return CosineAnnealingWarmRestarts(optimizer, T_0=params.pop("T_0", steps // 4),
                                               T_mult=params.pop("T_mult", 1),
                                               eta_min=params.pop("eta_min", 5e-5), **params)
        elif scheduler_type == "polynomial":
            return PolynomialLR(optimizer, total_iters=steps, power=params.pop("power", 1.0), **params)
        elif scheduler_type == "step":
            return StepLR(optimizer, step_size=params.pop("step_size", steps // 2),
                          gamma=params.pop("gamma", 0.1), **params)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    # ------------------------------------------------------------------
    # Data loader
    # ------------------------------------------------------------------

    def _init_dataloader(self) -> None:
        if self._dataset is None:
            data_sources = self._get_data_sources()
            self._dataset = PrecomputedDataset(
                self._config.data.preprocessed_data_root,
                data_sources=data_sources,
            )
            logger.debug(f"Loaded dataset with {len(self._dataset):,} samples from sources: {list(data_sources)}")

        dataloader = DataLoader(
            self._dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self._config.data.num_dataloader_workers,
            pin_memory=self._config.data.num_dataloader_workers > 0,
        )
        self._dataloader = self._accelerator.prepare(dataloader)

    # ------------------------------------------------------------------
    # Checkpointing and logging
    # ------------------------------------------------------------------

    def _save_checkpoint(self) -> Path:
        save_dir = Path(self._config.output_dir) / "checkpoints"
        save_dir.mkdir(exist_ok=True, parents=True)

        filename = f"model_weights_step_{self._global_step:05d}.safetensors"
        saved_weights_path = save_dir / filename
        rel_saved_weights_path = saved_weights_path.relative_to(self._config.output_dir)

        unwrapped_model = self._accelerator.unwrap_model(self._transformer)
        state_dict = unwrapped_model.state_dict()
        save_file(state_dict, saved_weights_path)
        logger.info(f"Model weights for step {self._global_step} saved in {rel_saved_weights_path}")

        if self._scalar_embedding is not None:
            scalar_filename = f"scalar_embedding_step_{self._global_step:05d}.safetensors"
            scalar_path = save_dir / scalar_filename
            unwrapped_scalar = self._accelerator.unwrap_model(self._scalar_embedding)
            save_file(unwrapped_scalar.state_dict(), scalar_path)
            logger.info(f"Scalar embedding weights for step {self._global_step} saved")

        self._checkpoint_paths.append(saved_weights_path)
        self._cleanup_checkpoints()
        return saved_weights_path

    def _cleanup_checkpoints(self) -> None:
        if 0 < self._config.checkpoints.keep_last_n < len(self._checkpoint_paths):
            checkpoints_to_remove = self._checkpoint_paths[: -self._config.checkpoints.keep_last_n]
            for old_checkpoint in checkpoints_to_remove:
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.debug(f"Removed old checkpoint: {old_checkpoint}")
            self._checkpoint_paths = self._checkpoint_paths[-self._config.checkpoints.keep_last_n :]

    def _save_epoch_log(self) -> None:
        if not IS_MAIN_PROCESS:
            return
        log_path = Path(self._config.output_dir) / "epoch_log.json"
        with open(log_path, "w") as f:
            json.dump(self._epoch_losses, f, indent=2)

    def _save_config(self) -> None:
        if not IS_MAIN_PROCESS:
            return
        config_path = Path(self._config.output_dir) / "training_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self._config.model_dump(), f, default_flow_style=False, indent=2)
        logger.info(f"Training configuration saved to: {config_path.relative_to(self._config.output_dir)}")

    def _setup_accelerator(self) -> None:
        self._accelerator = Accelerator(
            mixed_precision=self._config.acceleration.mixed_precision_mode,
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
        )
        if self._accelerator.num_processes > 1:
            logger.info(f"Distributed training enabled with {self._accelerator.num_processes} processes")
            effective_batch_size = (
                self._config.optimization.batch_size
                * self._accelerator.num_processes
                * self._config.optimization.gradient_accumulation_steps
            )
            logger.info(f"Global batch size: {effective_batch_size}")

    def _init_wandb(self) -> None:
        if not self._config.wandb.enabled or not IS_MAIN_PROCESS:
            self._wandb_run = None
            return
        wandb_config = self._config.wandb
        run = wandb.init(
            project=wandb_config.project,
            entity=wandb_config.entity,
            name=Path(self._config.output_dir).name,
            tags=wandb_config.tags,
            config=self._config.model_dump(),
        )
        self._wandb_run = run

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)

    @staticmethod
    def _log_training_stats(stats: TrainingStats) -> None:
        stats_str = (
            "Training Statistics:\n"
            f" - Total time: {stats.total_time_seconds / 60:.1f} minutes\n"
            f" - Training time: {stats.training_time / 60:.1f} minutes\n"
            f" - Training speed: {stats.steps_per_second:.2f} steps/second\n"
            f" - Samples/second: {stats.samples_per_second:.2f}\n"
            f" - Peak GPU memory: {stats.peak_gpu_memory_gb:.2f} GB"
        )
        if stats.compilation_time_seconds is not None:
            stats_str += f"\n - Compilation time: {stats.compilation_time_seconds:.1f} seconds"
        if stats.num_processes > 1:
            stats_str += f"\n - Number of processes: {stats.num_processes}"
            stats_str += f"\n - Global batch size: {stats.global_batch_size}"
        logger.info(stats_str)

    @staticmethod
    def _print_config(config: BaseModel) -> None:
        if not IS_MAIN_PROCESS:
            return

        from rich.table import Table

        table = Table(title="Training Configuration", show_header=True, header_style="bold green")
        table.add_column("Parameter", style="bold white")
        table.add_column("Value", style="bold cyan")

        def flatten_config(cfg: BaseModel, prefix: str = "") -> list[tuple[str, str]]:
            rows = []
            for field, value in cfg:
                full_field = f"{prefix}.{field}" if prefix else field
                if isinstance(value, BaseModel):
                    rows.extend(flatten_config(value, full_field))
                elif isinstance(value, (list, tuple, set)):
                    value_str = ", ".join(str(item) for item in value)
                    if len(value_str) > 70:
                        value_str = value_str[:70] + "..."
                    rows.append((full_field, value_str))
                else:
                    value_str = str(value)
                    if len(value_str) > 70:
                        value_str = value_str[:70] + "..."
                    rows.append((full_field, value_str))
            return rows

        skip_prefixes = []
        if hasattr(config, "scalar_conditioning") and not config.scalar_conditioning.enabled:
            skip_prefixes.append("scalar_conditioning.")

        for param, value in flatten_config(config):
            if any(param.startswith(p) for p in skip_prefixes):
                continue
            table.add_row(param, value)

        rich.print(table)
