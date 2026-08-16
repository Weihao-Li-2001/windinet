# Based on LTX-Video-Trainer by Lightricks (Apache 2.0).
# https://github.com/Lightricks/LTX-Video-Trainer
# Modified: added ScalarConditioningConfig, removed LoRA and text conditioning.
"""Pydantic configuration models for WinDiNet training and inference."""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

try:
    from windinet.inference.model_loader import LtxvModelVersion
except ImportError:
    LtxvModelVersion = None


class ConfigBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(ConfigBaseModel):
    """Configuration for the base model."""

    model_source: str | Path | LtxvModelVersion = Field(
        default=LtxvModelVersion.latest(),
        description="Model source - can be a HuggingFace repo ID, local path, or LtxvModelVersion",
    )

    load_checkpoint: str | Path | None = Field(
        default=None,
        description="Path to a checkpoint file or directory to load from. "
        "If a directory is provided, the latest checkpoint will be used.",
    )

    # noinspection PyNestedDecorators
    @field_validator("model_source", mode="before")
    @classmethod
    def validate_model_source(cls, v):  # noqa: ANN001, ANN206
        """Try to convert model source to LtxvModelVersion if possible."""
        if isinstance(v, (str, LtxvModelVersion)):
            try:
                return LtxvModelVersion(v)
            except ValueError:
                return v
        return v


class ConditioningConfig(ConfigBaseModel):
    """Configuration for conditioning during training."""

    first_frame_conditioning_p: float = Field(default=0.1, ge=0.0, le=1.0)


class ScalarConditioningConfig(ConfigBaseModel):
    """Configuration for scalar embeddings (e.g., inlet_speed, field_size)."""

    enabled: bool = Field(
        default=False,
        description="Whether to enable scalar conditioning",
    )

    scalar_names: list[str] = Field(
        default=["inlet_speed_mps", "field_size_m"],
        description="Names of scalars to embed",
    )

    scalar_ranges: dict[str, tuple[float, float]] = Field(
        default={
            "inlet_speed_mps": (0.1, 20.0),
            "field_size_m": (900.0, 1400.0),
        },
        description="Min/max ranges for each scalar (used for normalization to [0, 1])",
    )

    embedding_dim: int = Field(
        default=4096,
        description="Dimension of scalar embeddings (should match transformer hidden size)",
    )

    num_tokens_per_scalar: int = Field(
        default=4,
        description="Number of embedding tokens to generate per scalar",
        ge=1,
        le=16,
    )

    fourier_features: int = Field(
        default=64,
        description="Number of Fourier features for positional encoding of scalars",
        ge=8,
    )

    mlp_hidden_dim: int = Field(
        default=256,
        description="Hidden dimension of the MLP that processes scalar embeddings",
    )

    dropout: float = Field(
        default=0.0,
        description="Dropout probability for scalar embedding MLP",
        ge=0.0,
        le=1.0,
    )

    @field_validator("scalar_ranges")
    @classmethod
    def validate_scalar_ranges(cls, v: dict, info: ValidationInfo) -> dict:
        """Validate that ranges are provided for all scalar names."""
        scalar_names = info.data.get("scalar_names", [])
        for name in scalar_names:
            if name not in v:
                raise ValueError(f"Range must be provided for scalar '{name}'")
            min_val, max_val = v[name]
            if min_val >= max_val:
                raise ValueError(f"Invalid range for scalar '{name}': min ({min_val}) >= max ({max_val})")
        return v


class OptimizationConfig(ConfigBaseModel):
    """Configuration for optimization parameters."""

    learning_rate: float = Field(default=5e-4)
    steps: int = Field(default=3000)
    batch_size: int = Field(default=2)
    gradient_accumulation_steps: int = Field(default=1)
    max_grad_norm: float = Field(default=1.0)
    optimizer_type: Literal["adamw", "adamw8bit"] = Field(default="adamw")
    scheduler_type: Literal["constant", "linear", "cosine", "cosine_with_restarts", "polynomial"] = Field(
        default="linear"
    )
    scheduler_params: dict = Field(default_factory=dict)
    enable_gradient_checkpointing: bool = Field(default=False)


class AccelerationConfig(ConfigBaseModel):
    """Configuration for hardware acceleration and compute optimization."""

    mixed_precision_mode: Literal["no", "fp16", "bf16"] | None = Field(default="bf16")
    compile_with_inductor: bool = Field(default=True)
    compilation_mode: Literal["default", "reduce-overhead", "max-autotune"] = Field(default="reduce-overhead")


class DataConfig(ConfigBaseModel):
    """Configuration for data loading and processing."""

    preprocessed_data_root: str = Field(description="Path to folder containing preprocessed training data")
    num_dataloader_workers: int = Field(default=2, ge=0)


class ValidationConfig(ConfigBaseModel):
    """Held-out evaluation during DiT training.

    Disabled unless ``data_root`` points at a preprocessed split the training set
    does not contain (see ``preprocess_dataset.py --eval-sims``). Without it the
    run only ever reports training loss, which cannot distinguish learning from
    memorization.
    """

    data_root: str | None = Field(
        default=None,
        description="Preprocessed held-out split (a dir with latents/ and scalars/). "
        "None disables validation.",
    )
    interval: int | None = Field(
        default=None,
        gt=0,
        description="Optimizer steps between validation passes. Defaults to checkpoints.interval.",
    )
    max_samples: int = Field(
        default=0,
        ge=0,
        description="Cap on validation simulations per pass (0 = the whole split).",
    )


class CheckpointsConfig(ConfigBaseModel):
    """Configuration for model checkpointing during training."""

    interval: int | None = Field(default=None, gt=0)
    keep_last_n: int = Field(default=1, ge=-1)
    save_best_only: bool = Field(
        default=False,
        description=(
            "Keep a constant amount of disk instead of one checkpoint per epoch. Writes "
            "vae_shockwave_best.safetensors (overwritten only when best_metric improves) "
            "plus a rolling vae_shockwave_last.{safetensors,state.pt} pair for resuming. "
            "The 'best' weights carry no optimizer state on purpose: pairing best-epoch "
            "weights with last-epoch optimizer moments would silently corrupt a resume."
        ),
    )
    best_metric: Literal["val_total_loss", "val_vrmse"] = Field(
        default="val_total_loss",
        description=(
            "Metric that decides 'best' (lower is better). val_total_loss depends on the "
            "configured loss weights and so is not comparable across runs that weight the "
            "terms differently; val_vrmse is weight-independent."
        ),
    )
    save_last_state: bool = Field(
        default=True,
        description=(
            "save_best_only mode: also keep the rolling last/ pair so the run can resume. "
            "Set false to keep only the best weights (no resume possible)."
        ),
    )
    save_epochs: list[int] = Field(
        default_factory=list,
        description=(
            "Epochs whose weights are always force-saved as "
            "vae_shockwave_epoch{N:03d}_snapshot.safetensors, regardless of "
            "save_best_only / keep_last_n pruning. Use to preserve a specific "
            "epoch's weights for inspection even if it isn't the best or last epoch."
        ),
    )


class WandbConfig(ConfigBaseModel):
    """Configuration for Weights & Biases logging."""

    enabled: bool = Field(default=False)
    project: str = Field(default="windinet")
    entity: str | None = Field(default=None)
    tags: list[str] = Field(default_factory=list)


class FlowMatchingConfig(ConfigBaseModel):
    """Configuration for flow matching training."""

    timestep_sampling_mode: Literal["uniform", "shifted_logit_normal"] = Field(default="shifted_logit_normal")
    timestep_sampling_params: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# VAE decoder finetuning configs
# ---------------------------------------------------------------------------


class VaeReconstructionLossConfig(ConfigBaseModel):
    """Parameters for shockwave reconstruction loss components."""

    wavelet: str = Field(default="db2")
    spatial_level: int | None = Field(default=2, ge=1)
    temporal_level: int | None = Field(default=2, ge=1)
    mlw_beta: float = Field(default=10.0, ge=0.0)
    mlw_eps: float = Field(default=1e-6, gt=0.0)


class VaeDataConfig(ConfigBaseModel):
    """Dataset configuration for VAE decoder finetuning."""

    data_root: str = Field(description="Path to the shockwave HDF5 file")
    eval_sims: int = Field(default=10, description="Number of simulations held out for VRMSE evaluation")
    num_dataloader_workers: int = Field(default=4, ge=0)
    num_sim_frames: int | None = Field(default=None, ge=1, description="Optional frame limit for debugging")
    overfit_sims: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Diagnostic: train AND evaluate on the same first N simulations, ignoring "
            "eval_sims. Generalization is removed from the picture, so the VRMSE this "
            "reaches is the reconstruction floor imposed by the frozen encoder's latent "
            "for these exact samples. Never use for a real run."
        ),
    )
    overfit_repeat: int = Field(
        default=1,
        ge=1,
        description=(
            "overfit_sims only: how many times the subset is repeated per epoch. Sets the "
            "optimizer steps per epoch, since validation runs once per epoch and would "
            "otherwise dominate the runtime of a tiny training set."
        ),
    )
    channel_mean: list[float] | None = Field(
        default=None,
        description="Per-channel training-set means, positionally aligned with channel_order. "
        "Mutually exclusive with normalization_stats_file -- set exactly one of the two.",
    )
    channel_std: list[float] | None = Field(
        default=None,
        description="Per-channel training-set standard deviations, positionally aligned with "
        "channel_order. Mutually exclusive with normalization_stats_file -- set exactly one of the two.",
    )
    normalization_stats_file: str | Path | None = Field(
        default=None,
        description=(
            "Path to a YAML/JSON file with a data_normalization_stats block -- "
            "{channel_name: {mean, std, ...}}, the same shape "
            "configs/finetune_vae/euler_mq_128_only_train.yaml already has and "
            "scripts/compute_channel_stats.py's --output writes. When set, "
            "channel_mean/channel_std are populated from it (looked up by name and "
            "reordered to match channel_order) instead of being typed inline -- lets "
            "every config for a given resolution share one canonical stats file "
            "(and pick up a recomputed one, e.g. after scripts/compute_channel_stats.py "
            "runs against a new dataset like 256x256_ds, without hand-editing every "
            "config's numbers) instead of duplicating/hand-reindexing the same four "
            "numbers per config the way channel-order-sweep configs previously did."
        ),
    )
    normalization_clip: float = Field(
        default=5.0,
        gt=0.0,
        description="Map mean +/- this many standard deviations to [-1, 1]",
    )
    channel_order: list[str] = Field(
        default=["density", "momentum_x", "momentum_y", "pressure"],
        description=(
            "Which physical field is stacked into which of the 4 tensor channels, "
            "positionally aligned with channel_mean/channel_std (index i of both "
            "must refer to the same field) -- must stay identical to "
            "adapter.channels, which only carries the same order as metadata for "
            "logging/visualization. adapter.mode='inflate' always grows the VAE's "
            "conv_in/conv_out at index 3, so whichever field is last here is the "
            "one that gets a freshly-initialized (not pretrained-copied) slot -- "
            "see the channel-order sweep in EXPERIMENTS.md."
        ),
    )

    log_transform_channels: list[str] = Field(
        default_factory=list,
        description=(
            "Physical fields to replace with log(field) before z-score normalization "
            "(windinet.training.shockwave_data.build_shockwave_video), instead of the raw "
            "value. Restricted to the two strictly-positive fields (density, pressure) -- "
            "momentum_x/momentum_y can be negative, log() is undefined there. When paired "
            "with normalization_stats_file, the stats lookup automatically uses the "
            "'log_<name>' entry (e.g. euler_mq_128_only_train.yaml's log_density block) "
            "instead of '<name>' for these channels -- using inline channel_mean/channel_std "
            "instead means supplying already-log-space statistics yourself."
        ),
    )

    @field_validator("log_transform_channels")
    @classmethod
    def validate_log_transform_channels(cls, values: list[str]) -> list[str]:
        allowed = {"density", "pressure"}
        bad = sorted(set(values) - allowed)
        if bad:
            raise ValueError(
                f"log_transform_channels only supports strictly-positive fields "
                f"{sorted(allowed)}; got {bad} (momentum_x/momentum_y can be negative, "
                "log() is undefined there)"
            )
        if len(set(values)) != len(values):
            raise ValueError("log_transform_channels must not contain duplicates")
        return values

    @field_validator("channel_order")
    @classmethod
    def validate_channel_order(cls, values: list[str]) -> list[str]:
        expected = {"density", "momentum_x", "momentum_y", "pressure"}
        if len(values) != 4 or set(values) != expected:
            raise ValueError(f"channel_order must be a permutation of {sorted(expected)}, got {values}")
        return values

    @model_validator(mode="after")
    def validate_normalization_stats(self):
        has_inline = self.channel_mean is not None or self.channel_std is not None
        has_file = self.normalization_stats_file is not None
        if has_inline and has_file:
            raise ValueError(
                "set either channel_mean/channel_std or normalization_stats_file, not both"
            )
        if has_file:
            if self.channel_mean is not None or self.channel_std is not None:
                raise ValueError("normalization_stats_file set: channel_mean/channel_std must be unset")
            import yaml

            with open(self.normalization_stats_file) as fh:
                # yaml.safe_load also parses JSON (a YAML subset) -- same loader
                # works for the hand-written euler_mq_128_only_train.yaml and for
                # scripts/compute_channel_stats.py's --output JSON.
                stats = yaml.safe_load(fh).get("data_normalization_stats", {})
            # log_transform_channels entries look up "log_<name>" (e.g.
            # log_density) instead of "<name>" -- see build_shockwave_video,
            # which replaces these fields with log(field) before z-scoring,
            # so the z-score itself must use the log-space statistics.
            stats_keys = [
                f"log_{name}" if name in self.log_transform_channels else name
                for name in self.channel_order
            ]
            missing = [key for key in stats_keys if key not in stats]
            if missing:
                raise ValueError(
                    f"normalization_stats_file={self.normalization_stats_file!r} is missing "
                    f"stats for {missing} (needed for channel_order={self.channel_order}, "
                    f"log_transform_channels={self.log_transform_channels})"
                )
            self.channel_mean = [stats[key]["mean"] for key in stats_keys]
            self.channel_std = [stats[key]["std"] for key in stats_keys]
            # Clear the file reference now that its values are resolved inline --
            # otherwise a config dumped after this validator runs (e.g.
            # VaeTrainer._save_config's training_config.yaml) has BOTH fields set,
            # and reloading that dump re-triggers the has_inline-and-has_file error
            # above on what was a perfectly valid config a moment ago.
            self.normalization_stats_file = None
        elif self.channel_mean is None or self.channel_std is None:
            raise ValueError("set either channel_mean and channel_std, or normalization_stats_file")

        if len(self.channel_mean) != 4 or len(self.channel_std) != 4:
            raise ValueError("channel_mean and channel_std must each contain four values")
        if any(value <= 0 for value in self.channel_std):
            raise ValueError("all channel_std values must be positive")
        return self


class VaeOptimizationConfig(ConfigBaseModel):
    """Optimization for VAE decoder finetuning."""

    learning_rate: float = Field(default=5e-5)
    min_learning_rate: float = Field(
        default=1e-6,
        description=(
            "LR floor for the decoder group at the end of decay. Applied as the ratio "
            "min_learning_rate/learning_rate to every param group, so each group's LR "
            "multiplier is preserved for the whole schedule."
        ),
    )
    scheduler_type: Literal["cosine", "wsd", "constant", "warmup_plateau"] = Field(
        default="cosine",
        description=(
            "LR schedule after warmup. 'cosine' anneals to the floor over the whole run. "
            "'wsd' holds the peak LR for stable_fraction of the post-warmup steps, then "
            "cosine-anneals to the floor -- a flat val curve during the stable phase is "
            "evidence of a real plateau rather than an exhausted schedule. 'constant' "
            "never decays. 'warmup_plateau' holds the peak LR after warmup and decays by "
            "plateau_factor whenever checkpoints.best_metric stops improving for "
            "plateau_patience epochs -- the stable/decay split is discovered from this "
            "run's own validation curve instead of fixed in advance."
        ),
    )
    stable_fraction: float = Field(
        default=0.7,
        gt=0.0,
        le=1.0,
        description="wsd only: fraction of post-warmup steps held at peak LR before decay.",
    )
    plateau_factor: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description="warmup_plateau only: multiply LR by this factor on each plateau event.",
    )
    plateau_patience: int = Field(
        default=3,
        ge=1,
        description=(
            "warmup_plateau only: epochs with no improvement in checkpoints.best_metric "
            "before the LR is decayed."
        ),
    )
    plateau_threshold: float = Field(
        default=1e-4,
        gt=0.0,
        description="warmup_plateau only: minimum relative improvement to reset patience.",
    )
    epochs: int = Field(default=10)
    batch_size: int = Field(default=1)
    gradient_accumulation_steps: int = Field(default=32)
    max_grad_norm: float = Field(default=5.0)
    weight_decay: float = Field(default=0.0)
    warmup_steps: int = Field(default=50, description="Linear warmup optimizer steps")
    warmup_start_factor: float = Field(default=0.01)
    enable_gradient_checkpointing: bool = Field(default=True)


class VaeAdapterConfig(ConfigBaseModel):
    """Input/output channel adapters used while finetuning the VAE."""

    enabled: bool = Field(default=False, description="Wrap the VAE with trainable input/output adapters")
    checkpoint: str | Path | None = Field(default=None, description="Optional adapter/decoder checkpoint to resume from")
    channels: list[str] = Field(
        default=["density", "momentum_x", "momentum_y", "pressure"],
        min_length=1,
    )
    mode: Literal["adapter", "inflate"] = Field(
        default="adapter",
        description="'adapter': 1x1 in/out adapters around a frozen 3-ch VAE. 'inflate': grow the VAE's conv_in/conv_out to read/write all channels natively (trains encoder.conv_in too).",
    )
    inflate_init: Literal["zeros", "mean", "random", "copy"] = Field(
        default="zeros",
        description=(
            "mode='inflate' only. 'zeros' keeps every pretrained slot and zeroes the new "
            "channel (preserves the pretrained forward); 'mean' seeds the new channel with "
            "I3D-style averaging of the originals; 'copy' seeds the new channel with an exact "
            "copy of one specific original slot (see inflate_copy_channel) instead of "
            "averaging all of them -- for a physically-paired field (e.g. momentum_y as the "
            "new channel, momentum_x as an original one); 'random' discards the pretrained "
            "conv_in/conv_out entirely and reinitializes both, rescaled to preserve output "
            "variance -- the bet that LTXV's RGB patchify basis is wrong for CFD fields and "
            "is better relearned than adapted."
        ),
    )
    inflate_copy_channel: str | None = Field(
        default=None,
        description=(
            "inflate_init='copy' only: the physical channel name (must be one of the first "
            "3 entries of `channels`, i.e. an original pretrained-RGB slot, not the newly-"
            "grown 4th one) whose conv_in/conv_out weights are copied into the new channel's "
            "slot at init."
        ),
    )

    @model_validator(mode="after")
    def validate_inflate_copy_channel(self) -> "VaeAdapterConfig":
        if self.inflate_init == "copy":
            if self.inflate_copy_channel is None:
                raise ValueError("inflate_init='copy' requires inflate_copy_channel to be set")
            if self.inflate_copy_channel not in self.channels[:3]:
                raise ValueError(
                    f"inflate_copy_channel={self.inflate_copy_channel!r} must be one of the "
                    f"first 3 (original, pretrained-RGB) entries of channels={self.channels}, "
                    "not the newly-grown 4th one"
                )
        elif self.inflate_copy_channel is not None:
            raise ValueError("inflate_copy_channel is only meaningful when inflate_init='copy'")
        return self
    freeze_conv_in: bool = Field(
        default=False,
        description=(
            "mode='inflate' only: keep the grown encoder.conv_in frozen and train the decoder "
            "alone. encoder.conv_in is the last trainable thing on the encoder side, so freezing "
            "it fixes the latent space: already-encoded latents (and any DiT trained on them) stay "
            "valid, and the refined VAE can be swapped in at inference. Leave False when the "
            "latent space is still being established -- the extra channel cannot reach the encoder "
            "otherwise."
        ),
    )
    unfreeze_encoder_trunk: bool = Field(
        default=True,
        description=(
            "mode='inflate' only: also unfreeze the encoder's down_blocks[0:3] + tail "
            "(down_blocks[-1]+mid_block+norm_out+conv_out) -- the whole-structure unfreeze that "
            "is this config's default. False restricts encoder training to conv_in alone (see "
            "freeze_conv_in) -- decoder + conv_in only, matching the pre-whole-structure-unfreeze "
            "baseline. Ignored under mode='adapter': that mode's whole point is a frozen VAE "
            "behind trainable in/out adapters, so the encoder trunk is always frozen there "
            "regardless of this flag."
        ),
    )
    hidden_channels: int = Field(default=32, ge=1, description="Adapter hidden width (checkpoint metadata wins when resuming)")
    activation: Literal["relu", "silu", "swish", "gelu", "tanh"] = "gelu"
    identity_init: bool = Field(
        default=False,
        description="Init adapters as identity + zero residual (no tanh dead zone). Recommended for training from scratch.",
    )
    default_temb: float = Field(default=0.0)


# The four losses every VAE finetuning config has always had to weight.
REQUIRED_LOSS_NAMES = {"rmse", "h1", "ssim", "mlw"}
# Added 2026-08-06: computed by windinet.losses.reconstruction_losses like the
# required four, but default to an unset weight (0.0, via .get() at every
# summation site in vae_trainer.py) so no existing config needs updating to
# stay valid. Add one of these to loss_weighting.weights (fixed strategy) or
# loss_names (gradnorm/softadapt) to opt in. "kl" additionally needs
# VaeTrainer._encode's posterior mean/logvar to be non-None to actually do
# anything -- see reconstruction_losses' docstring. "anchor" (added
# 2026-08-16, windinet.losses.latent_anchor) similarly needs the rescaled
# latents, which VaeTrainer._forward_pass always supplies.
OPTIONAL_LOSS_NAMES = {"h2", "pcc", "vrms", "kl", "anchor"}


class LossWeightingConfig(ConfigBaseModel):
    """Composition strategy for the shockwave reconstruction losses."""

    strategy: Literal["fixed", "gradnorm", "softadapt"] = "fixed"
    weights: dict[str, float] = Field(
        default={"rmse": 1.0, "h1": 0.5, "ssim": 0.2, "mlw": 0.05},
    )
    loss_names: list[str] = Field(default=["rmse", "h1", "ssim", "mlw"])
    alpha: float = Field(default=1.5, gt=0.0)
    weight_lr: float = Field(default=0.025, gt=0.0)
    temperature: float = Field(default=0.1, gt=0.0)

    @model_validator(mode="after")
    def validate_loss_names(self):
        configured = set(self.weights if self.strategy == "fixed" else self.loss_names)
        missing = REQUIRED_LOSS_NAMES - configured
        if missing:
            raise ValueError(f"loss weighting must configure at least {sorted(REQUIRED_LOSS_NAMES)}; missing {sorted(missing)}")
        unknown = configured - REQUIRED_LOSS_NAMES - OPTIONAL_LOSS_NAMES
        if unknown:
            raise ValueError(
                f"loss weighting has unknown loss names {sorted(unknown)}; "
                f"known optional extras are {sorted(OPTIONAL_LOSS_NAMES)}"
            )
        return self


class VaeVisualizationConfig(ConfigBaseModel):
    """Periodic reconstruction plots for a fixed validation subset."""

    enabled: bool = Field(default=True)
    interval_epochs: int = Field(default=1, ge=1)
    num_samples: int = Field(
        default=3,
        ge=1,
        description=(
            "How many eval sims to render, chosen as an evenly-spaced spread over the eval "
            "set's gamma range rather than eval order -- 1 is the median gamma, 2 is "
            "min/max, 3 (default) is min/median/max, N>3 adds evenly-spaced points between "
            "the extremes. Selection is fixed per run (recomputed from eval_set, not "
            "random), so the same sims are plotted every epoch."
        ),
    )
    frame_numbers: list[int] = Field(default=[25, 50, 75, 100], min_length=1)
    dpi: int = Field(default=150, ge=72)

    @field_validator("frame_numbers")
    @classmethod
    def validate_frame_numbers(cls, values: list[int]) -> list[int]:
        if any(value < 1 for value in values):
            raise ValueError("visualization frame_numbers are one-based and must be >= 1")
        return values


class VaeTrainerConfig(ConfigBaseModel):
    """Configuration for shockwave VAE decoder-and-adapter finetuning."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    adapter: VaeAdapterConfig = Field(default_factory=VaeAdapterConfig)
    data: VaeDataConfig = Field(default_factory=VaeDataConfig)
    optimization: VaeOptimizationConfig = Field(default_factory=VaeOptimizationConfig)
    loss: VaeReconstructionLossConfig = Field(default_factory=VaeReconstructionLossConfig)
    loss_weighting: LossWeightingConfig = Field(default_factory=LossWeightingConfig)
    visualization: VaeVisualizationConfig = Field(default_factory=VaeVisualizationConfig)
    acceleration: AccelerationConfig = Field(default_factory=AccelerationConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    seed: int = Field(default=42)
    output_dir: str = Field(default="finetune_vae_outputs/vae_finetune")
    clean_output_dir: bool = Field(
        default=False,
        description="Delete output_dir before training so each run starts from a clean directory.",
    )
    resume_from: str | Path | None = Field(
        default=None,
        description="Path to a checkpoint .safetensors to resume from. Its finetuned "
        "weights are loaded into the model, and the sibling '<stem>.state.pt' "
        "(optimizer, scheduler, loss weights, RNG, epoch/step) is restored so training "
        "continues from the epoch after the one saved. Both files are read into memory "
        "before output_dir is (optionally) cleaned, so resuming into the same output_dir "
        "is safe.",
    )
    resume_weights_only: bool = Field(
        default=False,
        description=(
            "Load only the weights from resume_from and ignore its sibling .state.pt. "
            "Required for a warm restart: restoring the state would also restore the finished "
            "cosine schedule and the epoch counter, so a new learning_rate/epochs would be "
            "silently overridden and training would continue at the old run's final LR."
        ),
    )

    @field_validator("output_dir")
    @classmethod
    def expand_vae_output_path(cls, v: str) -> str:
        return str(Path(os.path.expandvars(v)).expanduser().resolve())

    @model_validator(mode="after")
    def validate_channel_order_consistency(self):
        if self.data.channel_order != self.adapter.channels:
            raise ValueError(
                "data.channel_order and adapter.channels must be identical lists -- "
                f"got data.channel_order={self.data.channel_order} vs "
                f"adapter.channels={self.adapter.channels}. data.channel_order controls "
                "the actual tensor stack order; adapter.channels is metadata (labels, "
                "checkpoint info, visualization titles) that must describe it correctly."
            )
        return self


# ---------------------------------------------------------------------------
# Diffusion model training configs
# ---------------------------------------------------------------------------


class LtxvTrainerConfig(ConfigBaseModel):
    """Unified configuration for WinDiNet training."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    conditioning: ConditioningConfig = Field(default_factory=ConditioningConfig)
    scalar_conditioning: ScalarConditioningConfig = Field(default_factory=ScalarConditioningConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    acceleration: AccelerationConfig = Field(default_factory=AccelerationConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    flow_matching: FlowMatchingConfig = Field(default_factory=FlowMatchingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    seed: int = Field(default=42)
    output_dir: str = Field(default="outputs")

    # noinspection PyNestedDecorators
    @field_validator("output_dir")
    @classmethod
    def expand_output_path(cls, v: str) -> str:
        return str(Path(os.path.expandvars(v)).expanduser().resolve())
