# Pretrained base weights

The LTX-Video weights WinDiNet builds on live here, in `pretrained/hub/`, instead
of in `~/.cache/huggingface/hub`. A checkout is therefore self-contained: nothing
outside the repo has to be present for training, preprocessing, or inference.

`pretrained/hub/` is a regular HuggingFace hub cache, so `huggingface_hub` reads
it offline and writes new downloads into it unchanged:

| Directory | Size | Used for |
| --- | --- | --- |
| `models--Lightricks--LTX-Video/` | 6.0 GB | `ltxv-2b-0.9.6-dev-04-25.safetensors` — the transformer for `model_source: LTXV_2B_0.9.6_DEV` |
| `models--Lightricks--LTX-Video-0.9.5/` | 2.4 GB | the VAE (0.9.6-DEV has no VAE of its own and falls back to the 0.9.5 repo) |
| `models--Lightricks--LTX-Video-0.9.7-dev/` | 32 KB | scheduler config only |

The contents are gitignored — 8.4 GB does not belong in git history.

## How the path is resolved

`windinet/paths.py` derives `HF_CACHE_DIR` from the package location, so it
follows the checkout wherever it sits. `import windinet` exports `HF_HUB_CACHE`
early (before `huggingface_hub` freezes its constants), and every load call in
`windinet/inference/model_loader.py` additionally passes `cache_dir=` explicitly.
The `jobs/**/*.sbatch` scripts export `HF_HUB_CACHE="${PWD}/pretrained/hub"` too,
which covers subprocesses that never import `windinet`.

Set `WINDINET_HF_CACHE=/path/to/hub` to point at a copy elsewhere (shared scratch,
a cluster-wide cache).

## Populating a fresh checkout

Copy from a checkout that already has them:

```bash
rsync -a --info=progress2 <host>:/path/to/windinet/pretrained/ pretrained/
```

Or let the loader download them (needs network, and `HF_HUB_OFFLINE` unset):

```bash
python -c "import windinet; from windinet.inference.model_loader import load_ltxv_components; load_ltxv_components('LTXV_2B_0.9.6_DEV')"
```

Finetuned VAE/DiT checkpoints are *not* stored here — they land in
`finetune_vae_outputs_*/<run>/checkpoints/` and are selected via
`WINDINET_VAE_ADAPTER_CKPT` / `WINDINET_VAE_INFLATE_CKPT`.
