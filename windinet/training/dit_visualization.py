"""Periodic GT-vs-prediction visualization panels for DiT training.

Mirrors windinet.training.vae_visualization's role for the VAE trainer, but
for DiT: samples a full flow-matching rollout from a fixed set of held-out
simulations (frame 0 as conditioning, same conditioning convention as
scripts/inference_shockwave.py), decodes the result through the VAE, and
reuses save_reconstruction_panels for the actual plotting -- static GT/
prediction/residual PNGs, not the MP4 video scripts/visualize_dit_predictions.py
renders (that renderer costs ~1-2 min/sample, too slow to run every few
hundred training steps; video stays a separate post-hoc step for the final
checkpoint, run that script by hand when needed).

IMPORTANT: this does NOT reuse LtxvTrainer's own self._vae. That VAE is
loaded from model.model_source (the generic pretrained 3-channel LTX VAE) as
plumbing DiT training itself never decodes through -- DiT trains entirely on
precomputed latents, so self._vae cannot even read the 4-channel shockwave
latents this trainer actually produces. The VAE that DID produce those
latents (an inflate-mode finetuned checkpoint) is recorded in
<preprocessed_data_root>/normalization.json's "vae_checkpoint" field (written
by preprocess_dataset.py); DitVisualizer loads that checkpoint separately,
once, and caches it for the life of the run -- same load_inflated_vae() used
by scripts/inference_shockwave.py.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import torch
from torch.amp import autocast

from windinet.inference.model_loader import load_inflated_vae
from windinet.inference.pipeline import LTXConditionPipeline
from windinet.training.shockwave_data import CHANNEL_NAMES, ShockWaveDataset, normalize_fields
from windinet.training.vae_visualization import denormalize_fields, save_reconstruction_panels
from windinet.utils import logger


def pick_fixed_visualization_sample_ids(
    preprocessed_data_root: str | Path, num_samples: int
) -> tuple[list[str], str]:
    """Evenly-spaced, deterministic val_ids picks + the raw HDF5 path.

    Reads <preprocessed_data_root>/split_manifest.json (written by
    preprocess_dataset.py) and returns the same fixed sample ids every call
    for the same (preprocessed_data_root, num_samples) pair -- this is the
    single source of truth for "which 3 samples" both `DitVisualizer` (the
    periodic in-training PNG panels) and scripts/inference_shockwave.py (the
    post-hoc video render, run separately/manually) use, so passing the same
    preprocessed_data_root and num_samples to both locks them onto the exact
    same simulations without either needing to record/pass sample ids
    explicitly.
    """
    manifest_path = Path(preprocessed_data_root) / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} not found -- re-run preprocess_dataset.py against this "
            "preprocessed_data_root first."
        )
    manifest = json.loads(manifest_path.read_text())
    val_ids = manifest.get("val_ids") or []
    if not val_ids:
        raise ValueError(f"{manifest_path} has no val_ids -- nothing to pick a fixed sample set from.")

    n = min(num_samples, len(val_ids))
    picks = sorted({round(i * (len(val_ids) - 1) / max(n - 1, 1)) for i in range(n)})
    picked_ids = [val_ids[i] for i in picks]
    return picked_ids, manifest["data_root"]


class DitVisualizer:
    """Lazily-built, cached GT-vs-prediction panel renderer for a fixed sample set.

    Construction is cheap (no model loading); the expensive setup (reading the
    manifest, opening the raw HDF5, loading the decode VAE) happens on the
    first call to `run`, not in `__init__`, so building this object doesn't
    cost anything for a run that never actually triggers a visualization pass.
    """

    def __init__(
        self,
        *,
        preprocessed_data_root: str,
        model_source,
        scalar_names: list[str],
        num_samples: int,
        frame_numbers: list[int],
        num_inference_steps: int,
        dpi: int,
        output_dir: str,
        seed: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._preprocessed_data_root = Path(preprocessed_data_root)
        self._model_source = model_source
        self._scalar_names = scalar_names
        self._num_samples = num_samples
        self._frame_numbers = frame_numbers
        self._num_inference_steps = num_inference_steps
        self._dpi = dpi
        self._output_dir = output_dir
        self._seed = seed
        self._device = device
        self._dtype = dtype

        self._samples: list[dict] | None = None  # lazy: fixed raw ShockWaveDataset rows
        self._stats: dict | None = None
        self._vae = None
        self._pipe: LTXConditionPipeline | None = None

    def _lazy_init(self) -> None:
        if self._samples is not None:
            return

        norm_path = self._preprocessed_data_root / "normalization.json"
        manifest_path = self._preprocessed_data_root / "split_manifest.json"
        if not norm_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"DiT visualization needs both {norm_path} and {manifest_path} "
                "(both written by preprocess_dataset.py) -- re-run preprocessing "
                "if either is missing, or set visualization.enabled: false."
            )

        payload = json.loads(norm_path.read_text())
        vae_checkpoint = payload.get("vae_checkpoint")
        if not vae_checkpoint:
            raise ValueError(f"{norm_path} has no vae_checkpoint recorded -- cannot decode DiT samples.")
        self._stats = {
            "channel_mean": payload["channel_mean"],
            "channel_std": payload["channel_std"],
            "normalization_clip": payload["normalization_clip"],
        }

        # Fixed, evenly-spaced picks across the held-out split, chosen once and
        # reused every call -- same "same samples every time" intent as VAE
        # viz's gamma-spread selection, just index-based: the manifest doesn't
        # carry per-sample gamma, and opening every held-out sample's h5 entry
        # just to sort by gamma isn't worth it for a 3-sample pick. Shared with
        # scripts/inference_shockwave.py's --preprocessed_data_root option, so
        # the post-hoc video render can lock onto these exact same samples.
        picked_ids, raw_h5 = pick_fixed_visualization_sample_ids(
            self._preprocessed_data_root, self._num_samples
        )
        dataset = ShockWaveDataset(raw_h5)
        id_to_idx = {sid: i for i, sid in enumerate(dataset.ids)}
        missing = [sid for sid in picked_ids if sid not in id_to_idx]
        if missing:
            raise ValueError(f"Visualization sample ids {missing} not found in {raw_h5}")
        self._samples = [dataset[id_to_idx[sid]] for sid in picked_ids]
        logger.info(
            f"DiT visualization fixed on {len(self._samples)} samples from {raw_h5}: "
            f"{[s['id'] for s in self._samples]}"
        )

        logger.info(f"Loading DiT-visualization VAE from {vae_checkpoint}")
        self._vae = load_inflated_vae(
            self._model_source, vae_checkpoint, dtype=self._dtype, device=str(self._device)
        )
        self._vae.requires_grad_(False)
        self._vae.eval()

    @torch.no_grad()
    def run(self, *, transformer, scalar_embedding, scheduler, step: int) -> None:
        """Sample + decode + save panels for the fixed samples.

        `transformer`/`scalar_embedding` must already be the unwrapped (non-DDP)
        modules in eval mode -- same discipline as LtxvTrainer._validate(), and
        the caller's responsibility, not this method's (it has no accelerator
        to unwrap with). Main-process-only by convention, same as checkpoint
        saving and validation -- not enforced here either, callers gate it.
        """
        self._lazy_init()

        if self._pipe is None:
            self._pipe = LTXConditionPipeline(
                scheduler=deepcopy(scheduler),
                vae=self._vae,
                text_encoder=None,
                tokenizer=None,
                transformer=transformer,
            ).to(self._device)
            self._pipe.set_progress_bar_config(disable=True)
            # See scripts/inference_shockwave.py's make_pipe() for why: our
            # fields are already in [-1, 1] via the CFD channel stats, and the
            # video processor's default output clamp would wipe out negative
            # momentum values.
            self._pipe.video_processor.register_to_config(do_normalize=False)
        else:
            self._pipe.transformer = transformer

        for i, sample in enumerate(self._samples):
            H, W = sample["density"].shape[-2:]
            gt = torch.stack([sample[name] for name in CHANNEL_NAMES]).unsqueeze(0)  # [1, C, F, H, W]
            num_frames_needed = gt.shape[2]
            num_frames_padded = ((num_frames_needed - 1) // 8 + 1) * 8 + 1  # LTX VAE needs 8k+1

            cond = normalize_fields(
                gt[:, :, 0],  # frame 0 as the initial condition, [1, C, H, W]
                self._stats["channel_mean"],
                self._stats["channel_std"],
                self._stats["normalization_clip"],
            ).unsqueeze(1).to(device=self._device, dtype=self._dtype)  # [1, 1, C, H, W]

            scalar_values = [sample["meta"][name] for name in self._scalar_names]
            scalars = torch.tensor([scalar_values], device=self._device, dtype=self._dtype)
            prompt_embeds = scalar_embedding(scalars)
            prompt_mask = torch.ones(1, prompt_embeds.shape[1], device=self._device, dtype=torch.long)

            g = torch.Generator(device=self._device).manual_seed(self._seed + i)
            with autocast(self._device.type, dtype=self._dtype, enabled=(self._device.type in ("cuda", "xpu"))):
                out = self._pipe(
                    prompt=None, negative_prompt=None,
                    video=cond, frame_index=0, strength=1.0,
                    width=W, height=H,
                    num_frames=num_frames_padded,
                    num_inference_steps=self._num_inference_steps,
                    guidance_scale=1.0,
                    image_cond_noise_scale=0.0,
                    generator=g,
                    output_reference_comparison=False,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_mask,
                    negative_prompt_embeds=torch.zeros_like(prompt_embeds),
                    negative_prompt_attention_mask=prompt_mask.clone(),
                    output_type="pt",
                )

            pred = out.frames[0].float().cpu()[:num_frames_needed]  # [F, C, H, W], trim the padding
            pred = pred.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, F, H, W]
            pred = denormalize_fields(
                pred, self._stats["channel_mean"], self._stats["channel_std"], self._stats["normalization_clip"],
            )

            save_reconstruction_panels(
                prediction=pred[0],
                target=gt[0],
                sample_id=sample["id"],
                label=f"step_{step:06d}",
                frame_numbers=self._frame_numbers,
                channel_names=CHANNEL_NAMES,
                output_dir=self._output_dir,
                dpi=self._dpi,
            )

            if self._device.type == "cuda":
                torch.cuda.empty_cache()
            elif self._device.type == "xpu":
                torch.xpu.empty_cache()

        logger.info(
            f"Saved DiT visualization panels for step {step} "
            f"({len(self._samples)} samples x {len(self._frame_numbers)} frames)"
        )
