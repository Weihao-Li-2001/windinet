# WinDiNet: Pretrained Video Models as Differentiable Physics Simulators for Urban Wind Flows

<p>
  <a href="https://arxiv.org/abs/2603.21210" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-2603.21210-b31b1b.svg" alt="arXiv Paper"/>
  </a>
  <a href="https://huggingface.co/rabischof/windinet" target="_blank">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97_Model-Hugging_Face-ffcc4d" alt="Hugging Face Model"/>
  </a>
  <a href="https://rbischof.github.io/windinet_web/" target="_blank">
    <img src="https://img.shields.io/badge/Web_Demo-Launch-2ea44f" alt="Web Demo"/>
  </a>
</p>

WinDiNet repurposes the [LTX-Video](https://github.com/Lightricks/LTX-Video) video diffusion transformer as a fast, differentiable surrogate for computational fluid dynamics (CFD) simulations of urban wind patterns. Fine-tuned on 10,000 CFD simulations, it generates full **112-frame velocity rollouts in under a second**.

## Installation

**Prerequisites:** Python 3.10+, CUDA-capable GPU (48 GB VRAM recommended for training).

```bash
pip install -e .

# For training (adds decord, pandas, scipy):
pip install -e ".[training]"

# For inverse design optimization (adds scipy):
pip install -e ".[inverse]"
```

## Checkpoints

Pretrained checkpoints are hosted on [HuggingFace](https://huggingface.co/rabischof/windinet) and **downloaded automatically** on first use. No manual setup is needed.

The repository contains three files:
- `dit.safetensors` -- Finetuned diffusion transformer (DiT)
- `scalar_embedding.safetensors` -- Scalar conditioning module
- `vae_decoder.safetensors` -- Physics-informed VAE decoder

To download manually:

```bash
huggingface-cli download rabischof/windinet --local-dir checkpoints/
```

## Inference

### Input format

Each sample consists of a pair of files in the input directory:
- `name.png` -- Building footprint image (black=building, white=fluid). Resized to 256x256 internally.
- `name.json` -- Scalar conditioning: `{"inlet_speed_mps": 10.0, "field_size_m": 1400}`

See `examples/footprints/` for sample inputs and `examples/predictions/` for corresponding outputs.

### Running

```bash
python scripts/inference.py configs/inference.yaml \
    --input_dir examples/footprints/
```

All three checkpoints are downloaded automatically on first run. Settings are in `configs/inference.yaml`. CLI flags (`--checkpoint`, `--num_inference_steps`, etc.) override the config.

### Output format

Each prediction is saved as `{name}.npz` and `{name}.mp4`:

**NPZ fields:**
- `u_fields`: horizontal velocity [T, H, W] in m/s (float16)
- `v_fields`: vertical velocity [T, H, W] in m/s (float16)
- `bldg_mask`: building footprint [H, W] (bool)

**MP4 video:** wind magnitude with coolwarm colormap.

## Training

WinDiNet training has two stages: (1) finetuning the VAE decoder with physics-informed losses, then (2) training the diffusion transformer with scalar conditioning.

### Stage 1: VAE decoder finetuning

Finetune the VAE decoder to improve reconstruction of wind velocity fields. The physics-informed loss enforces incompressibility and wall boundary conditions:

```bash
python scripts/finetune_vae.py configs/finetune_vae.yaml
```

Edit `configs/finetune_vae.yaml` to set `data.data_root` to your wind simulation dataset. The dataset should contain subdirectories, each with a `fields.npz` (keys: `u_fields`, `v_fields`, `bldg_mask`) and a `meta.json` (key: `wind_speed_mps`).

The loss function combines three terms (see `windinet/training/losses.py`):
- **Distance-weighted MSE**: reconstruction loss with higher weight near building boundaries
- **Divergence loss**: penalises violations of incompressibility (du/dx + dv/dy ~ 0)
- **Wall no-penetration loss**: enforces zero normal velocity at building walls

The resulting checkpoint is used at inference via `WINDINET_VAE_ADAPTER_CKPT`.

### Stage 2: Diffusion model training

#### Data preprocessing

Encode wind field simulations into VAE latents and extract scalar conditioning values:

```bash
python scripts/preprocess_dataset.py /path/to/wind_dataset/train \
    --output-dir /path/to/preprocessed
```

The script reads `fields.npz` + `meta.json` from each sample directory, truncates to 112 simulation frames, prepends a conditioning frame, encodes through the VAE, and saves latent tensors + scalars as `.pt` files.

#### Running training

```bash
python scripts/train.py configs/windinet.yaml
```

Edit `configs/windinet.yaml` to set `data.preprocessed_data_root` and `output_dir`.

## Metrics

```bash
python scripts/metrics.py \
    --pred_dir /path/to/predictions \
    --samples_root /path/to/gt_samples \
    --manifest /path/to/dataset.json \
    --out_dir /path/to/metrics
```

Outputs `per_sample.csv` and `summary.csv` with: vRMSE, MAE (m/s), MRE (%), MSE, Spectral Divergence, Wasserstein distance.

## Inverse Design

Optimise building layouts for pedestrian wind comfort using WinDiNet as a differentiable surrogate:

```bash
python scripts/inverse_design.py configs/inverse_opt.yaml
```

See `configs/inverse_opt.yaml` for all parameters. An example building layout is provided in `examples/inverse_optimization/`. The optimizer adjusts building positions to minimise a Pedestrian Wind Comfort (PWC) loss that penalises dangerous (>15 m/s), uncomfortable (>5 m/s), and stagnant (<1 m/s) wind conditions.

### Extending the inverse optimization

The inverse design framework is designed to be extensible in two directions:

**Custom objectives** (`inverse/objective.py`): Add new loss functions that operate on the predicted velocity fields (u, v) and a spatial mask. Any function returning a dict with a `"total"` key works as a drop-in replacement. For example, you could add building-code compliance checks, noise-based comfort metrics, or pollutant dispersion penalties. See the module docstring for the expected interface.

**Custom building parametrizations** (`inverse/footprint.py`): Replace the axis-aligned rectangle representation with arbitrary differentiable shapes (splines, polygons, level sets). Any `nn.Module` that returns a soft `(H, W)` occupancy map from its `forward()` method is compatible with the optimizer. See the module docstring for the required interface.

## Architecture

WinDiNet modifies LTX-Video in two ways:

1. **VAE Physics Adapter** (`windinet/vae_adapter.py`): The VAE decoder is finetuned with physics-informed losses (`windinet/training/losses.py`), improving reconstruction fidelity for wind velocity fields. Loaded at inference time via `WINDINET_VAE_ADAPTER_CKPT`.

2. **Scalar Conditioning** (`windinet/scalar_embeddings.py`): Replaces text conditioning with Fourier-feature-encoded scalar inputs (inlet speed, field size), enabling precise physical parameterization.

## Acknowledgements

Built on [LTX-Video-Trainer](https://github.com/Lightricks/LTX-Video-Trainer) by Lightricks, licensed under Apache 2.0.
