# WinDiNet — compute-resource figures

Measured 2026-08-26 on `lundquist` (2× RTX A6000 48 GB) via
[`scripts/resource_profile.py`](../scripts/resource_profile.py).
Dataset: `euler_mq_dataset/128x128_ds/train.h5` (4,500 sims, 10 gamma values,
4 fields × 101 frames).

Everything below is **measured or read off disk**, not estimated, except where
a row explicitly says "scaled". Dataset sizes for 256/512 are the real
published shard sizes from the HuggingFace dataset repo (`rha6696/euler_mq`),
obtained from the repo tree without downloading.

> **Basis for "data read":** file bytes on disk (82.0 GB for the 128 train
> split), not the sum of per-dataset compressed payloads (75.4 GB). The gap is
> HDF5 chunk-index and container metadata. The filesystem serves the former.

---

## The six proposal answers (at the current 128×128 baseline)

### 1. Average number of processes

One MPI/DDP rank per GPU or XPU tile, plus 4 dataloader worker processes per
rank. Single-node.

| Job | Cluster | Ranks | IO workers | OS processes | Eff. batch | Wall |
|---|---|---|---|---|---|---|
| VAE finetune (production) | sng_pvc | 8 | 32 | 40 | 32 | 12:00:00 |
| DiT training (production) | sng_pvc | 8 | 16 | 24 | 128 | 24:00:00 |
| VAE finetune (H100, 2 GPU) | lrz_ai | 2 | 8 | 10 | 32 | 08:00:00 |
| VAE finetune (H100, 4 GPU) | lrz_ai | 4 | 16 | 20 | 32 | 06:00:00 |

**Average across job types: 5.5 ranks / 23.5 OS processes.** Largest
configuration: 8 ranks / 40 processes. Multi-node is supported by the
launchers but not in production use (the one 2-node attempt, job 521396, hung
in rendezvous — see `infra.md`).

### 2. Average job memory (total over all nodes)

Measured at 2 ranks, scaled to production width by rank count. Host figures are
**PSS** (proportional set size), which divides pages shared between a rank and
its forked dataloader workers by the number of processes mapping them. Summing
RSS across ranks double-counts those pages and overstates the total by ~40%.

| Stage | host RAM (CPU), 2 ranks | GPU, 2 ranks | host RAM (CPU), 8 ranks | GPU, 8 ranks |
|---|---|---|---|---|
| VAE finetune | 11.7 GB | 45.2 GB (22.6 GB/GPU) | **~47 GB** | **~181 GB** (22.6 GB/GPU) |
| DiT training | 19.8 GB | 95.0 GB (47.5 GB/GPU) | **~79 GB** | **~380 GB** (47.5 GB/GPU) |

All figures are job totals summed over all ranks, not per rank; the
per-GPU value is given in parentheses. GPU figures are **reserved** memory
— the amount that must be free on each card, which is the number to
provision against. The smaller "live" requirement is in Q3.

GPU memory scales linearly with rank count (data parallelism, full model
replica per rank, no sharding). Host RAM scales slightly sublinearly, since
each rank shares pages with its forked dataloader workers.

### 3. Maximum memory per process

| | peak (provision this) | steady state | device (live) | device (reserved) |
|---|---|---|---|---|
| VAE rank (batch 4) | **8,600 MB** | ~2,500 MB | 21.5 GB | 22.6 GB |
| DiT rank | **11,650 MB** | ~3,000 MB | 45.2 GB | 47.5 GB |
| Preprocessing (1 proc) | 7,860 MB | ~1,600 MB | 3.4 GB | 3.4 GB |

Dataloader workers add a few hundred MB each.

**The host peak is transient, not a working set.** It is a high-water mark set
by two staging buffers that never coexist, with the steady training loop far
below both. Measured breakdown of one rank (fp32, as the trainer loads it):

| stage | RSS | Δ |
|---|---|---|
| interpreter | 0.01 GiB | |
| + `import torch` | 0.33 GiB | +0.32 |
| + CUDA context | 0.44 GiB | +0.12 |
| + `load_vae(dtype=float32)` | **5.33 GiB** | **+4.89** |
| + `.to(cuda)` | 1.58 GiB | −3.75 |
| + CPU checkpoint payload | 3.54 GiB | +1.95 |

1. **fp32 model materialization at startup (~4.9 GiB).**
   `vae_trainer.py:162` calls `load_vae(..., dtype=torch.float32)`. The
   safetensors file is bf16 and normally mmap'd — loading *as bf16* costs only
   +0.25 GiB because pages stay file-backed. Converting to fp32 forces real
   anonymous memory for the whole model. It is freed on `.to(cuda)`, but glibc
   retains ~1.1 GiB of the arena rather than returning it to the OS.
2. **Checkpoint payload at save time (~4.65 GiB).**
   `_build_checkpoint_payload` does `.detach().cpu().contiguous()` for every
   saved tensor into one dict — the whole 1.247 B-parameter fp32 checkpoint
   staged in RAM before `save_file` writes it.

Both are reducible in code if node RAM ever becomes the binding constraint:
load bf16 and cast after `.to(cuda)`; stream tensors to `save_file` instead of
building one dict. Neither is on the critical path today.

**Per-process memory is genuinely private** — USS ≈ PSS ≈ RSS within 2% at the
process level, so the peak cannot be discounted as shared mmap pages. The
sharing that PSS corrects for is between a rank and its dataloader workers,
which is why it only shows up in the tree total (Q2), not per process.

**Read the two device columns correctly.** "Live" is
`torch.cuda.max_memory_allocated()` — the true high-water mark of live tensors,
and the real requirement. "Reserved" is the caching allocator's pool as the
driver reports it; on an otherwise-empty card the allocator expands to fill
available space, so the DiT's 47.5 GB reserved on a 47.54 GiB card does **not**
mean it needs 47.5 GB. The DiT's actual requirement is **45.2 GB**, leaving
~2.3 GiB of headroom on a 48 GB card.

That headroom is real but thin: the DiT stage needs an **exclusive** 48 GB GPU.
A probe run that shared a card with an unrelated 6.2 GB job OOM'd; the identical
run on an idle card completed at 45.24 GB. 80 GB-class cards (H100/A100) would
give genuine headroom, and are required if the DiT is ever run at higher
latent resolution (see Caveats).

### 4. Total data transferred to/from

| Direction | Item | Size |
|---|---|---|
| IN | Euler-MQ dataset (10 HDF5 shards) | 82.0 GB |
| IN | LTX-Video 2B pretrained weights | 8.2 GB |
| IN | Source checkout | 30 MB |
| **Total inbound** | once at project setup | **90 GB** |
| OUT | Diagnostics mirror (20 runs) | 3.1 GB |
| OUT | Promoted VAE checkpoints | 13.9 GB |
| OUT | Final DiT checkpoint + rollouts | 7.7 GB |
| **Total outbound** | over the campaign | **~25 GB** |

Outbound is small because every launcher mirrors results back with
`rsync -a --exclude='checkpoints/'` — weights stay on cluster scratch.

**Resident on project storage: ~408 GB** (82 GB dataset + 279 GB of 20 runs'
checkpoints + 36 GB DiT window + 8.2 GB weight cache + 3.1 GB diagnostics).

### 5. Frequency and size of data output/input

| Stream | Frequency | Size per event | Aggregate |
|---|---|---|---|
| HDF5 read (VAE) | every epoch × 30 | 17.2 MB/sim compressed (25.2 MB decompressed) | 82 GB/epoch, **2.40 TB/run** |
| VAE checkpoint | every epoch | 4.65 GB weights + 9.29 GB optimizer state | 418 GB written into 2 rewritten slots |
| Reconstruction panels | every epoch, rank 0 | 12 PNGs × ~450 kB | 158 MB |
| Latent preprocessing | once per promoted checkpoint | reads 82 GB, writes 57.8 kB + 1.8 kB per sim | 262 MB corpus |
| Latent read (DiT) | every step, 16 samples × 10,000 steps | 57.8 kB | 8.8 GB (whole corpus fits in page cache) |
| DiT checkpoint | every 60 steps (166 writes) | 7.17 GB | **1.16 TB of write traffic**, 35.8 GB resident |
| Slurm stdout/stderr | continuous | ~1 line/step | < 5 MB/job |

I/O is **bursty, not continuous** — checkpoint writes are the only large events.

### 6. Files and sizes in a typical production run

**One VAE run (30 epochs): 368 files / 18.7 GB**

| File | Count | Size each |
|---|---|---|
| `checkpoints/vae_shockwave_{best,last}.safetensors` | 2 | 4.65 GB |
| `checkpoints/vae_shockwave_last.state.pt` | 1 | 9.29 GB |
| `visualizations/epoch_####/<sim>/frame_####.png` | 360 | ~450 kB |
| `metrics/metrics.csv`, `loss_curves.png`, `training_config.yaml`, logs | 5 | < 200 kB |

**Preprocessing: 9,002 files / 262 MB** — `latents/<id>.pt` (57.8 kB) and
`scalars/<id>.pt` (1.8 kB), one pair per simulation. This is the only large
*file count* in the project.

**One DiT run (10,000 steps): 14 files / 35.9 GB** — a rolling window of 5 ×
7.17 GB transformer checkpoints plus 5 × 16.4 MB scalar-embedding checkpoints.

**Whole campaign (20 VAE runs + 1 DiT run): ~16,400 files, ~411 GB resident.**

---

## Extrapolation to 256×256 and 512×512

**Unchanged by resolution:** process counts (Q1), file counts and checkpoint
sizes (Q6). The VAE's trainable parameter count is fixed by the architecture,
not the input resolution, so a checkpoint is 4.65 GB at every resolution and
the ~1.57 TB of per-pipeline checkpoint write traffic is identical.

**Scales with pixel count:** everything in Q4/Q5.

| | 128×128 | 256×256 | 512×512 (native) |
|---|---|---|---|
| Train set (real) | 82.0 GB | 320.0 GB | 1.23 TB |
| Per simulation | 18.7 MB | 72.8 MB | 286.5 MB |
| Read per epoch | 82 GB | 320 GB | 1.23 TB |
| Per 30-epoch run | 2.40 TB | 9.37 TB | **36.9 TB** |
| Total inbound | 90 GB | 328 GB | 1.24 TB |
| Latent grid | 4×4×14 (224 tok) | 8×8×14 (896 tok) | 16×16×14 (3,584 tok) |
| Latent `.pt` | 57.8 kB | 231 kB | 924 kB |
| Latent corpus | 262 MB | 1.02 GB | 4.1 GB |

The 256/512 ratios (3.90× and 15.35×) come in just under the 4×/16× pixel
ratio — gzip does marginally better on the higher-resolution fields.

### Device memory does *not* scale with pixel count

| per rank, live tensors | batch 1 | batch 4 |
|---|---|---|
| 128×128 | 21.55 GB | 21.55 GB |
| 256×256 | 21.55 GB | 30.48 GB |
| 512×512 | 29.69 GB | **OOM on 48 GB** |

Host RSS per rank is flat across all three (~7.3–8.2 GB).

**Why**, from a probe that isolates activations from everything else (single
rank, batch 1, bf16, gradient checkpointing on, 2.33 GB resident weights
excluded):

| | activations alone | vs 128 |
|---|---|---|
| 128×128 | 1.09 GB | 1.0× |
| 256×256 | 2.44 GB | 2.2× |
| 512×512 | 9.72 GB | 8.9× |

The trainer's high-water mark is the **optimizer step** — fp32 master weights
+ gradients + two AdamW moments for the ~900M trainable parameters of the
whole-structure unfreeze — not the activations. At 128 and 256 the activation
term sits entirely underneath that floor, which is why the end-to-end number
looks flat. At 512 activations finally exceed it: the +8.1 GB step from 21.55
to 29.69 GB matches the measured activation delta (9.72 − 1.09 = 8.6 GB).

### Practical consequence for a 512×512 campaign

`batch_size=4` — the current sng_pvc production default — measures 30.48 GB at
256² but **OOMs at 512²** on a 48 GB card (CUDA needed 1.57 GiB more than was
free). A 512² campaign needs either `batch_size=1` or 80 GB-class GPUs
(H100/A100). This is consistent with the OOM history in `infra.md`.

---

## Caveats

- 256/512 memory figures are measured against **synthetic datasets of
  identical shape** (upsampled from real 128 sims) — device memory depends on
  tensor shape, not values. Only `128x128_ds` is on local disk.
- All memory measured on **RTX A6000 at 2 ranks**; 8-rank totals are scaled by
  rank count, not measured.
- The DiT stage was only measured at 128² latents (224 tokens). At 256²/512²
  the latent sequence grows 4×/16×, which will raise DiT activation memory —
  **not yet measured**.
- Per-epoch wall-clock and total GPU-hours are **not** covered here; see
  `infra.md` for the throughput ledger.
