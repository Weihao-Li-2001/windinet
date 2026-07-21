# Based on LTX-Video-Trainer by Lightricks (Apache 2.0).
# https://github.com/Lightricks/LTX-Video-Trainer
# Modified: added ScalarConditioningConfig, removed LoRA and text conditioning.
"""Pydantic configuration models for WinDiNet training and inference."""

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


class CheckpointsConfig(ConfigBaseModel):
    """Configuration for model checkpointing during training."""

    interval: int | None = Field(default=None, gt=0)
    keep_last_n: int = Field(default=1, ge=-1)


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
    channel_mean: list[float] = Field(description="Per-channel training-set means")
    channel_std: list[float] = Field(description="Per-channel training-set standard deviations")
    normalization_clip: float = Field(
        default=5.0,
        gt=0.0,
        description="Map mean +/- this many standard deviations to [-1, 1]",
    )

    @model_validator(mode="after")
    def validate_normalization_stats(self):
        if len(self.channel_mean) != 4 or len(self.channel_std) != 4:
            raise ValueError("channel_mean and channel_std must each contain four values")
        if any(value <= 0 for value in self.channel_std):
            raise ValueError("all channel_std values must be positive")
        return self


class VaeOptimizationConfig(ConfigBaseModel):
    """Optimization for VAE decoder finetuning."""

    learning_rate: float = Field(default=5e-5)
    min_learning_rate: float = Field(default=1e-6, description="Minimum LR for cosine annealing")
    epochs: int = Field(default=10)
    batch_size: int = Field(default=1)
    gradient_accumulation_steps: int = Field(default=32)
    max_grad_norm: float = Field(default=5.0)
    weight_decay: float = Field(default=0.0)
    warmup_steps: int = Field(default=50, description="Linear warmup optimizer steps")
    warmup_start_factor: float = Field(default=0.01)
    vis_every_steps: int = Field(default=0, description="Visualize GT|Reconstruction|Error every N optimizer steps (0=disabled)")
    enable_gradient_checkpointing: bool = Field(default=True)


class VaeAdapterConfig(ConfigBaseModel):
    """Input/output channel adapters used while finetuning the VAE."""

    enabled: bool = Field(default=False, description="Wrap the VAE with trainable input/output adapters")
    checkpoint: str | Path | None = Field(default=None, description="Optional adapter/decoder checkpoint to resume from")
    channels: list[str] = Field(
        default=["density", "momentum_x", "momentum_y", "pressure"],
        min_length=1,
    )
    hidden_channels: int = Field(default=32, ge=1, description="Adapter hidden width (checkpoint metadata wins when resuming)")
    activation: Literal["relu", "silu", "swish", "gelu", "tanh"] = "gelu"
    default_temb: float = Field(default=0.0)


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
        expected = {"rmse", "h1", "ssim", "mlw"}
        configured = set(self.weights if self.strategy == "fixed" else self.loss_names)
        if configured != expected:
            raise ValueError(f"loss weighting must configure exactly {sorted(expected)}")
        return self


class VaeTrainerConfig(ConfigBaseModel):
    """Configuration for shockwave VAE decoder-and-adapter finetuning."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    adapter: VaeAdapterConfig = Field(default_factory=VaeAdapterConfig)
    data: VaeDataConfig = Field(default_factory=VaeDataConfig)
    optimization: VaeOptimizationConfig = Field(default_factory=VaeOptimizationConfig)
    loss: VaeReconstructionLossConfig = Field(default_factory=VaeReconstructionLossConfig)
    loss_weighting: LossWeightingConfig = Field(default_factory=LossWeightingConfig)
    acceleration: AccelerationConfig = Field(default_factory=AccelerationConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    seed: int = Field(default=42)
    output_dir: str = Field(default="outputs/vae_finetune")

    @field_validator("output_dir")
    @classmethod
    def expand_vae_output_path(cls, v: str) -> str:
        return str(Path(v).expanduser().resolve())


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
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    flow_matching: FlowMatchingConfig = Field(default_factory=FlowMatchingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    seed: int = Field(default=42)
    output_dir: str = Field(default="outputs")

    # noinspection PyNestedDecorators
    @field_validator("output_dir")
    @classmethod
    def expand_output_path(cls, v: str) -> str:
        return str(Path(v).expanduser().resolve())
