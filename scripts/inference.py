#!/usr/bin/env python3
"""
WinDiNet -- Batch inference for wind field prediction.

Uses scalar conditioning (inlet speed + field size) to generate 112-frame
velocity rollouts. Saves predictions as .npz (u_fields, v_fields, bldg_mask)
in m/s (float16).

Input: a directory of samples, each consisting of:
    - A building footprint PNG (black=building, white=fluid, 256x256)
    - A JSON file with inlet_speed_mps and field_size_m

Usage:
    WINDINET_VAE_ADAPTER_CKPT=checkpoints/vae_physics.safetensors \\
    python scripts/inference.py configs/inference.yaml \\
        --input_dir examples/footprints/ \\
        --out_dir predictions/
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from safetensors.torch import load_file
from torch.amp import autocast

from windinet.checkpoints import ensure_checkpoint
from windinet.inference.pipeline import LTXConditionPipeline
from windinet.inference.model_loader import load_ltxv_components
from windinet.scalar_embeddings import ScalarEmbedding
from windinet.config import ScalarConditioningConfig
from windinet.visualization import render_wind_video

W, H = 256, 256
DTYPE = torch.bfloat16


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path, help="Path to inference YAML config")
    ap.add_argument("--input_dir", type=Path, required=True,
                    help="Directory with footprint PNGs and matching JSON files")
    ap.add_argument("--out_dir", type=Path, default=Path("outputs/inference"),
                    help="Output directory for .npz predictions")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Override transformer checkpoint path")
    ap.add_argument("--scalar_checkpoint", type=Path, default=None,
                    help="Override scalar embedding checkpoint path")
    ap.add_argument("--num_inference_steps", type=int, default=None,
                    help="Override number of denoising steps")
    ap.add_argument("--guidance_scale", type=float, default=None,
                    help="Override guidance scale")
    return ap.parse_args()


def load_config(args):
    """Load YAML config and apply CLI overrides."""
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.checkpoint is not None:
        cfg["checkpoint"] = str(args.checkpoint)
    if args.scalar_checkpoint is not None:
        cfg["scalar_checkpoint"] = str(args.scalar_checkpoint)
    if args.num_inference_steps is not None:
        cfg["num_inference_steps"] = args.num_inference_steps
    if args.guidance_scale is not None:
        cfg["guidance_scale"] = args.guidance_scale

    return cfg


def png_to_bmask(png_path: Path) -> np.ndarray:
    """Load a footprint PNG and return a 256x256 boolean mask (True=building)."""
    img = Image.open(png_path).convert("L")
    img = img.resize((W, H), Image.NEAREST)
    return np.array(img) < 128


def make_cond_image(speed_mps: float, bmask: np.ndarray, mag_cap: float) -> Image.Image:
    """Build RGB conditioning frame: R=u, G=v, B=fluid mask."""
    u_n = np.clip(np.where(bmask, 0.0, speed_mps) / mag_cap, -1.0, 1.0)
    v_n = np.zeros((H, W), dtype=np.float32)
    r01 = (u_n + 1.0) * 0.5
    g01 = (v_n + 1.0) * 0.5
    b01 = (~bmask).astype(np.float32)
    rgb = np.clip(np.stack([r01, g01, b01], axis=-1) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def make_pipe(model_source, device):
    c = load_ltxv_components(
        model_source=model_source,
        transformer_dtype=DTYPE,
        vae_dtype=DTYPE,
    )
    pipe = LTXConditionPipeline(
        scheduler=deepcopy(c.scheduler),
        vae=c.vae,
        text_encoder=None,
        tokenizer=None,
        transformer=c.transformer,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    return pipe


def load_transformer_weights(pipe, checkpoint):
    print(f"Loading transformer: {checkpoint}")
    sd = load_file(str(checkpoint))
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    if any(k.startswith("transformer.") for k in sd):
        sd = {k.replace("transformer.", "", 1): v for k, v in sd.items() if k.startswith("transformer.")}
    pipe.transformer.load_state_dict(sd, strict=False)


def load_scalar_embedding(checkpoint, scalar_cfg, device):
    print(f"Loading scalar embedding: {checkpoint}")
    emb = ScalarEmbedding(scalar_cfg)
    emb.load_state_dict(load_file(str(checkpoint)))
    return emb.to(device=device, dtype=DTYPE).eval()


def save_rgb_video(rgb, out_path, fps=12):
    """Save raw RGB composite video (model output before velocity conversion)."""
    import imageio.v3 as iio
    frames = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    iio.imwrite(str(out_path), frames, fps=fps)


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    global DTYPE
    if device.type == "mps":
        DTYPE = torch.float16

    # Load settings from config
    model_source = cfg.get("model_source", "LTXV_2B_0.9.6_DEV")
    checkpoint = ensure_checkpoint(cfg["checkpoint"])
    scalar_checkpoint = ensure_checkpoint(cfg["scalar_checkpoint"])
    num_inference_steps = cfg.get("num_inference_steps", 2)
    guidance_scale = cfg.get("guidance_scale", 1.0)
    num_frames = cfg.get("num_frames", 113)
    seed = cfg.get("seed", 42)
    mag_cap_mps = cfg.get("mag_cap_mps", 30.0)

    # VAE decoder checkpoint (via env var or config)
    import os
    vae_ckpt = cfg.get("vae_checkpoint")
    if vae_ckpt and not os.getenv("WINDINET_VAE_ADAPTER_CKPT"):
        os.environ["WINDINET_VAE_ADAPTER_CKPT"] = ensure_checkpoint(vae_ckpt)

    # Build scalar conditioning config
    sc = cfg.get("scalar_conditioning", {})
    scalar_cfg = ScalarConditioningConfig(
        enabled=True,
        scalar_names=sc.get("scalar_names", ["inlet_speed_mps", "field_size_m"]),
        scalar_ranges={k: tuple(v) for k, v in sc.get("scalar_ranges", {
            "inlet_speed_mps": [0.1, 20.0],
            "field_size_m": [900.0, 1400.0],
        }).items()},
        embedding_dim=sc.get("embedding_dim", 4096),
        num_tokens_per_scalar=sc.get("num_tokens_per_scalar", 4),
    )

    # Discover samples
    samples = []
    for png_path in sorted(args.input_dir.glob("*.png")):
        json_path = png_path.with_suffix(".json")
        if not json_path.exists():
            print(f"WARNING: no JSON for {png_path.name}, skipping")
            continue
        meta = json.loads(json_path.read_text())
        samples.append((png_path.stem, png_path, meta))

    if not samples:
        print(f"No samples found in {args.input_dir} (expected *.png + *.json pairs)")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pipe = make_pipe(model_source, device)
    load_transformer_weights(pipe, checkpoint)
    scalar_emb = load_scalar_embedding(scalar_checkpoint, scalar_cfg, device)

    print(f"Generating {len(samples)} predictions -> {args.out_dir}  [steps={num_inference_steps}]")

    for i, (name, png_path, meta) in enumerate(samples):
        out_path = args.out_dir / f"{name}.npz"

        if out_path.exists():
            print(f"[{i+1}/{len(samples)}] {name} — exists, skipping")
            continue

        bmask = png_to_bmask(png_path)
        inlet_speed = float(meta["inlet_speed_mps"])
        field_size = float(meta["field_size_m"])
        cond_img = make_cond_image(inlet_speed, bmask, mag_cap_mps)

        scalars = torch.tensor([[inlet_speed, field_size]], device=device, dtype=DTYPE)
        prompt_embeds = scalar_emb(scalars)
        prompt_mask = torch.ones(1, prompt_embeds.shape[1], device=device, dtype=torch.long)

        g = torch.Generator(device=device).manual_seed(seed + i)
        use_dtype = DTYPE if device.type == "cuda" else torch.float32
        with autocast(device.type, dtype=use_dtype, enabled=(device.type != "mps")):
            out = pipe(
                prompt=None, negative_prompt=None,
                image=cond_img, width=W, height=H,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=g,
                output_reference_comparison=False,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_mask,
                negative_prompt_embeds=torch.zeros_like(prompt_embeds),
                negative_prompt_attention_mask=prompt_mask.clone(),
                output_type="pt",
            )

        rgb = out.frames[0].float().cpu().numpy()
        rgb = rgb.transpose(0, 2, 3, 1)
        u_fields = (rgb[..., 0] * 2.0 - 1.0) * mag_cap_mps
        v_fields = (rgb[..., 1] * 2.0 - 1.0) * mag_cap_mps
        u_fields, v_fields = u_fields[1:], v_fields[1:]

        np.savez_compressed(
            out_path,
            u_fields=u_fields.astype(np.float16),
            v_fields=v_fields.astype(np.float16),
            bldg_mask=bmask,
        )

        mp4_path = args.out_dir / f"{name}.mp4"
        render_wind_video(u_fields, v_fields, bmask, mp4_path)

        rgb_path = args.out_dir / f"{name}_rgb.mp4"
        save_rgb_video(rgb[1:], rgb_path)

        print(f"[{i+1}/{len(samples)}] {name} — saved ({u_fields.shape[0]} frames, "
              f"speed={inlet_speed:.1f} m/s, field={field_size:.0f} m)")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nDone! {len(samples)} .npz files -> {args.out_dir}")


if __name__ == "__main__":
    main()
