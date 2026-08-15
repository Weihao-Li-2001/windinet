# WinDiNet: Pretrained Video Models as Differentiable Physics Simulators

<p>
  <a href="https://arxiv.org/abs/2603.21210" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-2603.21210-b31b1b.svg" alt="arXiv Paper"/>
  </a>
</p>

WinDiNet repurposes the [LTX-Video](https://github.com/Lightricks/LTX-Video) video diffusion transformer as a fast, differentiable surrogate for computational fluid dynamics (CFD) simulations. This fork adapts the original urban-wind-flow model (2-channel u/v velocity + building mask, 256x256) to **ShockWaveNet**: 4-channel compressible Euler CFD fields (density, momentum_x, momentum_y, pressure) with shocks, 128x128, conditioned on the scalar `gamma`.

For experiment status, results, and open questions, see [EXPERIMENTS.md](EXPERIMENTS.md) -- that file, not this one, is the source of truth for what has actually been run and what it showed.

## Installation

**Prerequisites:** Python 3.10+, CUDA-capable GPU (48 GB VRAM recommended for training).

```bash
pip install -e .

# For training (adds decord, pandas, scipy):
pip install -e ".[training]"
```

### Base weights

By default, LTX-Video weights download to `huggingface_hub`'s own cache
(`~/.cache/huggingface`) like any other project -- no setup needed. Set
`WINDINET_HF_CACHE=/path/to/dir` (`windinet/paths.py`) to redirect to a
different location instead -- e.g. lundquist's `jobs/lundquist/*.sbatch`
launchers point it at an in-repo `pretrained/hub/` (~8.4 GB, gitignored) so
a checkout there is self-contained. **This is opt-in per machine, not a
repo-wide default** -- sng_pvc and lrz_ai deliberately leave it unset and
use the ordinary `~/.cache/huggingface` location (lrz_ai's `$HOME` is
DSS-quota'd and can't absorb an in-repo copy). See
[pretrained/README.md](pretrained/README.md) for the layout and how to
populate one.

## Training

Training has two stages: (1) finetuning the VAE to reconstruct the 4-channel CFD fields, then (2) training the diffusion transformer (DiT) on the resulting latents.

### Stage 1: VAE finetuning

```bash
python scripts/finetune_vae.py configs/finetune_vae/finetune_vae_baseline.yaml
```

Edit `data.data_root` in the config to point at your shockwave HDF5 dataset (`<sample_id>/{density,momentum_x,momentum_y,pressure}`, see `windinet/training/shockwave_data.py` for the expected layout). Checkpoints, per-epoch reconstruction panels, metrics and the resolved config are written under `output_dir`:

```
<output_dir>/
    checkpoints/vae_shockwave_{best,last,epoch###}.safetensors
    checkpoints/*.state.pt        # optimizer/scheduler/RNG, for resume_from
    visualizations/epoch_####/    # GT/prediction/residual panels
    metrics/{metrics.csv,loss_curves.png}
    training_config.yaml
```

Cluster launchers (Slurm): `jobs/lundquist/finetune_vae_{2,4,6}gpu.sbatch`, `jobs/sng_pvc/finetune_vae.sbatch`. Per-cluster storage/worker-count defaults live in `windinet/cluster_config.py`.

### Stage 2: Data preprocessing

Encode the CFD fields into VAE latents for DiT training:

```bash
python scripts/preprocess_dataset.py /path/to/shockwave_dataset/train.h5 \
    --output-dir /path/to/preprocessed \
    --inflate-checkpoint <output_dir>/checkpoints/vae_shockwave_best.safetensors \
    --eval-sims 675
```

This must use the *finetuned* VAE checkpoint from stage 1 -- see `scripts/preprocess_dataset.py`'s docstring for the exact output layout (`latents/`, `scalars/`, `normalization.json`, and a `val/` split when `--eval-sims` is set).

### Stage 3: DiT training

```bash
python scripts/train.py configs/shockwavenet.yaml
```

Set `data.preprocessed_data_root` (and `validation.data_root` to the `val/` split from preprocessing) and `output_dir` in the config. DiT training consumes precomputed latents only -- it never re-runs the VAE encoder -- and records which VAE checkpoint produced them in `<output_dir>/latent_provenance.json` for later verification.

Cluster launchers: `jobs/lundquist/train_dit.sbatch`, `jobs/sng_pvc/train_dit.sbatch`.

## Inference

```bash
python scripts/inference_shockwave.py configs/inference_shockwave.yaml \
    --h5 euler_mq_dataset/128x128_ds/train.h5 \
    --out_dir predictions/ --num_samples 8
```

Conditions on the initial condition (frame 0 of a simulation) plus scalar `gamma`, and rolls out the remaining frames as 4-channel `.npz` fields. The VAE checkpoint in the config must be the exact one the DiT's latents were encoded with -- the script refuses to decode (`verify_latent_space`) if the stored latent-space fingerprint doesn't match, since a mismatched decoder silently produces physically wrong output.

## Architecture

WinDiNet modifies LTX-Video in two ways:

1. **VAE channel adapter** (`windinet/vae_adapter.py`): grows the pretrained 3-channel encoder/decoder to 4 channels (`inflate` mode) so the CFD fields pass through natively, then finetunes the decoder (and `encoder.conv_in`) with reconstruction losses (`windinet/losses/`, weighted via `windinet/loss_weighting/`).
2. **Scalar conditioning** (`windinet/scalar_embeddings.py`): replaces text conditioning with Fourier-feature-encoded scalar inputs (currently just `gamma`), enabling physical parameterization instead of prompts.

## Acknowledgements

Built on [LTX-Video-Trainer](https://github.com/Lightricks/LTX-Video-Trainer) by Lightricks, licensed under Apache 2.0.
