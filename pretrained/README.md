# Pretrained base weights

**Only in use on lundquist.** By default (every other machine, including
sng_pvc and lrz_ai) weights download to `huggingface_hub`'s own cache,
`~/.cache/huggingface/hub` -- nothing here applies unless `WINDINET_HF_CACHE`
is explicitly set, which only `jobs/lundquist/*.sbatch` do. This directory
exists so a lundquist checkout is self-contained there: nothing outside the
repo has to be present for training, preprocessing, or inference on that
machine specifically. sng_pvc and lrz_ai must keep using the ordinary
`~/.cache/huggingface` location -- lrz_ai's `$HOME` is DSS-quota'd and can't
absorb an in-repo copy (job 5750198, "Disk quota exceeded", 2026-08-15,
from an earlier version of this setup that redirected every cluster).

`pretrained/hub/` is a regular HuggingFace hub cache, so `huggingface_hub` reads
it offline and writes new downloads into it unchanged:

| Directory | Size | Used for |
| --- | --- | --- |
| `models--Lightricks--LTX-Video/` | 6.0 GB | `ltxv-2b-0.9.6-dev-04-25.safetensors` — the transformer for `model_source: LTXV_2B_0.9.6_DEV` |
| `models--Lightricks--LTX-Video-0.9.5/` | 2.4 GB | the VAE (0.9.6-DEV has no VAE of its own and falls back to the 0.9.5 repo) |
| `models--Lightricks--LTX-Video-0.9.7-dev/` | 32 KB | scheduler config only |

The contents are gitignored — 8.4 GB does not belong in git history.

## How the path is resolved

`windinet/paths.py`'s `HF_CACHE_DIR` is `None` -- and `import windinet` a
no-op for cache purposes -- unless `WINDINET_HF_CACHE` is set in the
environment; every load call in `windinet/inference/model_loader.py`
passes `cache_dir=` explicitly, so it also does nothing without that env
var (would pass the string `"None"` otherwise, hence the explicit guard in
that module). `jobs/lundquist/*.sbatch` export `HF_HUB_CACHE="${PWD}/pretrained/hub"`
directly at the shell level (not via `WINDINET_HF_CACHE`), which covers
subprocesses that never import `windinet` too -- this is the only place
that happens; `jobs/sng_pvc/*.sbatch` and `jobs/lrz_ai/*.job` deliberately
do not.

Set `WINDINET_HF_CACHE=/path/to/hub` on a machine that should also use an
in-repo (or shared) cache to point `windinet.paths`/`model_loader.py` at it.

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
