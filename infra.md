# Cluster Infra Notes

Hardware/throughput/cluster-operational knowledge split out of
`EXPERIMENTS.md` (2026-08-17 reorganization) to keep that file about
modeling questions, not compute infrastructure. This covers: per-GPU
`batch_size` vs wall-clock (production defaults per cluster), the sng_pvc
dataloader/eval/rendezvous throughput investigation, and where job
logs/configs/outputs live on disk.

Nothing below was reworded when it moved; content is verbatim from
`EXPERIMENTS.md`'s prior form.

## Table of contents

- [Per-GPU batch_size vs wall-clock/OOM (Open Question 23, closed)](#per-gpu-batch_size-vs-wall-clockoom)
- [sng_pvc throughput diagnostic](#sng_pvc-throughput-diagnostic)
- [Eval parallelization fix, CONFIRMED WORKING](#eval-parallelization-fix-confirmed-working-2026-08-06)
- [Where things live](#where-things-live)

## Per-GPU batch_size vs wall-clock/OOM (Open Question 23, closed 2026-08-16)

23. ~~Does per-GPU micro-batch size (`optimization.batch_size`) change
    results or only wall-clock/OOM feasibility?~~ **CLOSED (2026-08-16):
    larger batch_size is faster with no val_vrmse cost clearing the "real
    effect" bar, at every effective_batch=32 arm actually run. Production
    batch_size adopted per cluster (lundquist=16, sng_pvc=4, lrz_ai=16 --
    all accum=1, the effective_batch=32 ceiling at each cluster's
    production GPU count); no further batch_size testing planned.**
    Results:
    | cluster | arms run | fastest vs `batch_size=1` | val_vrmse delta | verdict |
    |---|---|---|---|---|
    | sng_pvc (8 tiles) | 1, 2, 4 | `batch_size=4`: **-8%** wall-clock (job 524577 vs 524575) | +0.56% (inside ~1.1% noise floor) | adopt 4 |
    | lundquist (2 GPU) | 1, 16 | `batch_size=16`: **-18.7%** wall-clock (job 21991 vs 21989, 6.66h vs 8.19h/18ep) | +1.53% (above the 1.1% floor, inside the 2.2% "real effect" bar) | adopt 16 |
    | lrz_ai (1 GPU) | none completed | -- (all 5 arms crashed on an f-string `SyntaxError` in the launcher, fixed 2026-08-16 but never resubmitted) | -- | production value (16, 2-GPU) extrapolated from lundquist, not lrz_ai-confirmed |

    **Update (2026-08-17): the "not lrz_ai-confirmed" caveat above resolved
    negatively at 256res, not positively.** `jobs/lrz_ai/finetune_vae_2gpu.job`
    hardcoded `batch_size=16` (its own header already flagged this as an
    extrapolation from lundquist's 128res/A6000 result, untested at this
    script's own 256res default) -- two of five 256res launches at that
    setting (jobs 5750941/5750942, plain/`_kl1e7`) completed epoch 1 fine
    then hit a CUDA OOM early in epoch 2 (free memory down to ~1.7GB out of
    ~93GB total by then). The other three 256res arms in that same batch
    (`_kl1e6`, `_kl1e5`, `_anchor`) *did* complete 26-29 epochs at
    `batch_size=16` without incident, so this isn't "16 is unsafe at 256res"
    across the board -- more likely fragmentation/accumulation that some
    runs hit and others didn't by epoch 2. Fixed by making `BATCH_SIZE` an
    overridable env var (default dropped to 4, a conservative 4x-headroom
    guess, itself not yet confirmed) instead of hardcoding 16 -- see that
    script's own header. lrz_ai's production `batch_size` therefore still
    has no clean confirmed number at 256res; 16 remains the confirmed value
    at 128res only (via the lundquist extrapolation, still not lrz_ai-run
    directly).

    `lundquist`'s eval time (~66s/epoch) was identical regardless of
    `batch_size` -- the entire wall-clock difference is in the train phase,
    consistent with fewer/bigger forward-backward passes being the
    mechanism, not anything eval-side.

    **Decision on the batch_size x accum grid follow-up (effective_batch >
    32, designed 2026-08-16, `jobs/sng_pvc/finetune_vae_batchsize_grid.sbatch`):
    dropped, not pursued.** That follow-up would have isolated
    `effective_batch`'s own effect (holding `batch_size` fixed, varying only
    `accum`/`effective_batch`) -- still an open theoretical question (see
    the mechanism discussion below) -- but the user decided the
    effective_batch=32 arms above already answer the practically load-
    bearing question (what production `batch_size` to run at) well enough
    to close the topic without it. `finetune_vae_batchsize_grid.sbatch` and
    the three effective_batch=32 sweep launchers below are retired
    (removed, not archived -- same precedent as
    `jobs/lundquist/finetune_vae_6gpu.sbatch`, 69558ed) since none of them
    have further use once this question is closed; their results remain
    valid and citable by job ID above regardless.

    **THE QUESTION (kept for context on why the sweep was designed this
    way):** every run in this ledger so far uses `batch_size: 1`,
    with `gradient_accumulation_steps` doing all the work of reaching
    `effective_batch` (32, `windinet/cluster_config.py`'s cluster default)
    -- many small, serialized micro-steps per optimizer step. Larger
    `batch_size` at the same `effective_batch` changes nothing about the
    optimizer trajectory *in principle*: total optimizer steps and the LR
    schedule are derived from `effective_batch`/`n_train`, not `batch_size`
    (`patch_config_for_cluster` re-derives `gradient_accumulation_steps` so
    `num_processes x batch_size x accum` always equals 32, holding steps-
    per-epoch at ~120 regardless of the split -- per-rank micro-batches
    scale as `1913/batch_size`, accum as `16/batch_size`, their ratio
    constant). What changes is (a) how many samples sit in GPU memory at
    once per forward/backward -- an OOM risk larger `batch_size`
    introduces that accumulation never did -- and (b) how much of the
    effective batch is assembled via true parallelism (one bigger kernel
    launch) vs sequential accumulation, which could shift wall-clock either
    direction (less per-step Python/comm overhead vs worse memory-bandwidth
    utilization).
    **HYPOTHESIS:** val_vrmse should be indistinguishable across arms (same
    optimizer trajectory); the real questions are (1) which `batch_size`
    values survive without OOM on each cluster's GPU, and (2) how wall-clock
    actually moves.
    **Sweep:** `batch_size` in {1 (reference), 2, 4, 8, 16},
    `finetune_vae_whole_structure_baseline.yaml` (18ep) held fixed
    otherwise, split across two clusters (decided 2026-08-16, supersedes the
    original lundquist-first plan below):
    - **sng_pvc, 8 XPU tiles** (the tile count already used for job 523577,
      not a new single-tile setup) -- but `effective_batch=32` at 8 tiles
      caps `batch_size` at **4** (`batch_size x num_processes` must divide
      32; 8 and 16 would need `effective_batch` >= 64/128, a different
      experiment). Covers `batch_size` in {1, 2, 4} only.
      `jobs/sng_pvc/finetune_vae_batchsize.sbatch`.
    - **lrz_ai, 1 GPU (H100)** -- single rank, no DDP split, so it covers
      the full {1, 2, 4, 8, 16} range. `jobs/lrz_ai/finetune_vae_batchsize.job`.

    `batch_size=1` on sng_pvc reproduces job 523577's exact setup (same 8
    tiles, accum=4) as a harness sanity check; lrz_ai's `batch_size=1` arm
    is this config's first run on that cluster, no prior reference point.
    lundquist (2 GPUs, A6000, script written but not submitted) remains
    available if a third cluster's read-out is wanted later, since GPU
    memory capacity differs by cluster and a size that fits on one may not
    fit on another.
    **Read-out:** per arm, (a) OOM or not, (b) final val_vrmse vs the
    `batch_size=1` reference (bar: >2.2%, 2x the noise floor, would be
    surprising and worth explaining), (c) wall-clock/per-epoch timing vs
    the reference. `batch_size=1` also fills in lundquist's own missing
    reference point for this config -- previously only run on sng_pvc (job
    523577, val_vrmse 0.08360).
    **Decision rule (adopt as new cluster default once read out):** among
    arms that (a) don't OOM and (b) land within 2.2% val_vrmse of the
    `batch_size=1` reference, pick the one with the lowest mean per-epoch
    `total=` wall-clock from the log (`Epoch N timing: ... total=Xs`,
    `windinet/training/vae_trainer.py:983-985` -- average epochs 2+ to drop
    epoch-1 warmup/compile). If two or more arms land within 5% of each
    other's wall-clock (noise), prefer the *smaller* `batch_size`: it leaves
    more activation-memory headroom for the upcoming 256/512res configs
    (`finetune_vae_whole_structure_baseline_256res.yaml` and beyond), where
    the same `batch_size` will cost substantially more memory per sample
    than it does at 128x128. Do not just take the largest surviving
    `batch_size` on speed alone -- OOM headroom at this resolution says
    nothing about headroom at 4x the pixel count.
    **Kill criterion:** none in the usual train_loss sense -- OOM itself
    *is* the informative outcome for an arm here, not a bug to abort on.
    Only a non-memory failure (train_loss diverging, unrelated crash) would
    count as a real bug.
    **Launched (historical record):** sng_pvc `batch_size` in {1, 2, 4}
    (jobs 524575/524576/524577) and lundquist `batch_size` in {1, 16}
    (jobs 21989/21991, `batch_size=2` job 21990 cancelled by hand early,
    no data) both completed and are read out in the results table above.
    lrz_ai's 5 arms ({1,2,4,8,16}) all crashed on the launcher's f-string
    bug (fixed 2026-08-16) and were never resubmitted after the fix --
    lrz_ai's production `batch_size` above is extrapolated, not
    lrz_ai-confirmed (see the results table). The launcher scripts that ran
    these (`jobs/{sng_pvc,lundquist}/finetune_vae_batchsize.{sbatch}`,
    `jobs/lrz_ai/finetune_vae_batchsize.job`) and the never-launched grid
    follow-up (`jobs/sng_pvc/finetune_vae_batchsize_grid.sbatch`) are
    retired (removed 2026-08-16) now that this question is closed -- see
    "Decision on the batch_size x accum grid follow-up" above.

## sng_pvc throughput diagnostic

**THE PUZZLE:** sng_pvc job 520211 (`finetune_vae_baseline.yaml`, 8 Intel XPU
tiles) averaged 38.7 min/epoch -- roughly the SAME as lundquist's 2-GPU
(A6000) estimate of 32-36 min/epoch, despite 4x the raw compute. Something
was eating the expected speedup.

**METHOD:** `windinet/training/vae_trainer.py` now logs a per-epoch phase
breakdown (`train=`/`eval=`/`viz=`/`ckpt=`/`total=`), so a 2-epoch throwaway
run (epoch 1 = warmup/compile, epoch 2 = clean reading) isolates where the
time goes without re-reading a full job. Three single-variable diagnostics,
`jobs/sng_pvc/finetune_vae_diag.sbatch` (+ `finetune_vae_diag_4rank.sbatch`
for the 4th):

| exp | job | change | epoch-2 train | epoch-2 eval | epoch-2 total | verdict |
|---|---|---|---|---|---|---|
| 0 (base) | 520300 | none (`num_dataloader_workers: 0`, full ckpt+viz) | 1426.8s | 1002.1s | 2445.4s (40.8 min) | reference |
| 1 (H1: I/O-bound step) | 520301 | `num_dataloader_workers: 0 -> 2` | 872.3s | 1045.5s | 1933.0s (32.2 min) | **train time -39%, total -21%** -- confirmed |
| 2 (H3: ckpt/viz I/O to DSS) | 520302 | `save_last_state: false`, `visualization.enabled: false` | 1441.9s | 1002.1s | 2445.4s (40.8 min, unchanged) | ckpt+viz was only ~16s/epoch of the base run (`viz=11.7s ckpt=4.3s`) -- **ruled out**, not the bottleneck |
| 3 (H2: comms topology) | 520303 | 4 ranks (COMPOSITE) x accum 8, vs 8 ranks (FLAT) x accum 4 | -- | -- | **crashed every rank** | `RuntimeError: Invalid mt19937 state` -- untested, see fix below |
| 4 (H1 cont.: workers=4) | 520456 | `num_dataloader_workers: 2 -> 4` | 869.3s | 921.1s | 1805.9s (30.1 min) | **-6.6% total vs workers=2, no crash** -- best confirmed so far |
| 5 (H1 cont.: workers=8) | 520457 | `num_dataloader_workers: 2 -> 8` | 866.8s | 923.3s | 1809.6s (30.2 min) | statistically same as workers=4 -- no further gain past 4 |
| 6 (H2 retry, post-fix) | 520459 | 4 ranks (COMPOSITE), `rng_types=[]` fix applied | -- | -- | **hit time limit mid-epoch-1** | no mt19937 crash this time, but no timing data either -- H2 still open, needs a longer time limit |

**Exp 3 crash and fix:** all 3 non-rank-0 processes died with `Invalid
mt19937 state` inside `accelerate/utils/random.py`'s `synchronize_rng_state`.
Root cause: Accelerate's default `rng_types=["generator"]` broadcasts the
train-loader sampler's RNG state from rank 0 to every other rank on every
`accelerator.prepare()` call; that broadcast corrupts in transit through
oneCCL specifically under `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE` (4-rank) --
the identical 8-tile FLAT launches (exp 0-2) hit the same code path every
epoch without issue. Fixed in `windinet/training/vae_trainer.py`
(`_setup_accelerator`): `rng_types=[]`, safe because `train()` calls
Accelerate's `set_seed(cfg.seed)` identically on every rank *before* the
DataLoader/sampler is built, and the train/eval split uses its own
`cfg.seed`-derived `torch.Generator`, not the global RNG -- so every rank's
default sampler already gets an identical seed without any cross-rank
broadcast; the sync was redundant (and buggy) in this codepath. **Exp 3 has
not been re-run since the fix** -- H2 (communication topology) is still
open.

**ESTABLISHED:** `num_dataloader_workers: 0` on sng_pvc (`windinet/cluster_config.py`
`CLUSTER_DEFAULTS["sng_pvc"]`) was a crash-avoidance choice, not a
performance-tuned one -- the comment there notes 4 workers x 8 ranks crashed
opening `train.h5` concurrently on the DSS network filesystem. `workers: 2`
is confirmed safe at 8 ranks and cuts wall-clock per epoch by ~21%
(train-step time by ~39%) with no code change beyond the config's
`data.num_dataloader_workers` field. **Promoted to the cluster default**
(`CLUSTER_DEFAULTS["sng_pvc"]["num_dataloader_workers"] = 2`) -- every future
sng_pvc launch (including the head-vs-tail unfreeze sweep in Open Question 3)
gets this for free with no per-config change.

**Promoted further (2026-08-06):** exp 4-5 above show `workers: 4` shaves
another ~6.6% off `workers: 2` (30.1 vs 32.2 min/epoch, mostly from eval time
-- 921.1s vs 1045.5s -- not train, which was flat 869.3s vs 872.3s) with no
crash, and `workers: 8` gains nothing more past 4. Both diagnostics were only
2 epochs, so the transient-open-failure risk the old N=4 comment warned about
was never confirmed ruled out over a full 15-epoch run before this promotion
-- `CLUSTER_DEFAULTS["sng_pvc"]["num_dataloader_workers"]` is now **4**
(`windinet/cluster_config.py`), a deliberate decision to accept that
unconfirmed risk rather than spend a dedicated full-length run just to
de-risk it first. **Watch the first few post-change full runs' `.err` logs**
for the `train.h5`-concurrent-open failure mode the old N=4 comment
described; if it recurs, drop back to 2 and note it here.

**Correction (2026-08-06, later same day):** the "-6.6%, mostly from eval
time" reading above is likely **not actually caused by `num_dataloader_workers`**.
Before the "Eval parallelization fix" below, `eval_loader` was constructed
as `DataLoader(eval_set, batch_size=1, shuffle=False)` with no `num_workers`
argument at all -- it silently ignored `cfg.data.num_dataloader_workers` and
always ran with 0 workers, in both the `workers: 2` and `workers: 4`
diagnostics. The 921.1s vs 1045.5s eval-time gap between those two jobs was
therefore almost certainly ordinary run-to-run filesystem-contention noise
(consistent with the up-to-32% eval-time drift *within a single run*
documented in the `--time` paragraph below), not a real effect of the
`workers` config. `train` time being flat (869.3s vs 872.3s) across the same
two jobs is consistent with this: `train_loader` *does* read
`num_dataloader_workers`, and going 2->4 barely moved it. Net: the
`workers: 2 -> 4` promotion above still stands (it's free and not shown to
hurt), but don't expect the ~6.6% total-time win it was promoted on --
expect closer to 0, since eval (the larger of the two time sinks, ~41% of
total) was never actually reading this setting. The real eval fix is below.

Same date: `jobs/sng_pvc/finetune_vae.sbatch`'s `--time` was cut from
`24:00:00` to `12:00:00` (first set to `10:00:00`, then raised to `12:00:00`
before any job was submitted under the new default -- extra margin against
the eval-time-drift risk below). Every full-data 15-epoch run observed so
far finishes in 7h25m-8h54m at `workers: 2` (see the head-vs-tail sweep
table above); `workers: 4` should only shrink that further, so 12h leaves
comfortable margin even at the slowest observed runs (`fullenc`/`tail3`,
8h54m at `workers: 2`). Note `finetune_vae_baseline_tail3`'s own per-epoch
log showed eval time drifting up over the run (793s at epoch 6 -> 1048s at
epoch 15, train flat throughout) for reasons not yet understood (filesystem
contention from other jobs is the leading guess, not confirmed) -- worth
keeping an eye on if a run ever gets close to the limit despite this
margin. Shorter `--time` should also mean shorter queue wait, which is part
of the reason for the change; if a future config (e.g. a slower encoder-LR
sweep arm) still times out at 12h, raise it back up for that config
specifically rather than reverting the shared default.

**OPEN:** H2 (does 4-rank COMPOSITE vs 8-rank FLAT communication topology
matter?) -- job 520459 hit the diagnostic's time limit before finishing even
epoch 1 (the `rng_types=[]` fix itself worked, no crash), so re-run
`jobs/sng_pvc/finetune_vae_diag_4rank.sbatch` now that
`rng_types=[]` unblocks it. **Deprioritized** -- not being chased right now.

**H4 (node-count scaling): BLOCKED on rendezvous, diagnosed, fix applied but
UNTESTED (2026-08-06).** Does requesting 2 nodes (16 XPU tiles) train faster
than 1 node (8 tiles), or does crossing the inter-node fabric eat the extra
compute? Hypothesis: if communication was already close to the bottleneck
intra-node, going inter-node only makes it worse. Script:
`jobs/sng_pvc/finetune_vae_diag_2node.sbatch` (2-epoch diagnostic, same
`DIAG_TAG`/`DIAG_WORKERS`/`DIAG_EPOCHS` knobs as `finetune_vae_diag.sbatch`;
`patch_config_for_cluster` halves `gradient_accumulation_steps`
automatically to hold effective_batch=32 at 16 ranks). **Read-out:**
compare its epoch-2 `train=` time to job 520456's 869.3s (1 node, workers=4,
the fastest confirmed 1-node reference). **Kill criterion:** `train= >= 90%`
of 869.3s means 2-node scaling isn't worth chasing further (don't try 4
nodes next).

**First attempt, job 521396, failed before reaching epoch 1** -- this was
the first multi-node launch attempted on sng_pvc for this project, and the
rendezvous logic `finetune_vae.sbatch`/`finetune_vae_diag_2node.sbatch`
already carried for it (MASTER_ADDR = first node in `$SLURM_NODELIST`,
`--num_machines`/`--machine_rank` via srun's per-node `SLURM_PROCID`) had
never actually been exercised past `--nodes=1` before this:

```
torch.distributed.DistStoreError: wait timeout after 900000ms, keys:
/none/torchelastic/role_info/0, /none/torchelastic/role_info/1
```

**Diagnosis.** The `.err` log's c10d socket trace is the key evidence:

```
[W805 01:44:10...] waitForInput: poll for socket addr=[i20r02c04s06...]:39582, remote=[i20r02c04s04...]:45361 returned 0, likely a timeout
[W805 01:44:10...] ...timed out after 900000ms
[W805 01:44:11...] waitForInput: poll for socket addr=[i20r02c04s04...]:47330, remote=[i20r02c04s04...]:45361 returned 0, likely a timeout
[W805 01:44:11...] ...timed out after 900000ms
```

The first pair is the cross-node connection (node 1 -> node 0's rendezvous
store). The second pair is node 0 connecting to **its own** store --
loopback, never leaving the host. Both completed their TCP handshake
(there's a live `fd`/local+remote address on each) and then both hung
identically waiting for a response. Loopback hanging the same way as the
cross-node connection rules out an inter-node firewall/routing problem --
loopback traffic can't be blocked by a firewall between nodes -- and points
at the TCPStore **server** on the master never responding to any client,
local or remote. That matches a known failure signature of PyTorch's
libuv-backed TCPStore (the default since this torch version's `USE_LIBUV`
defaults to `"1"`; confirmed by reading `torch/distributed/rendezvous.py`
and `torch/distributed/elastic/utils/distributed.py` in the installed
package), which has reported hangs on some HPC/interconnect setups where
the older, pre-libuv TCPStore implementation works fine.

Also present in the same `.err`, but judged **not** the cause: `slurmstepd:
error: Unable to create TMPDIR [/hppfs/scratch/0D/go76fuz2/tmp]` (falls
back to per-node local `/tmp`). This is emitted by `slurmstepd` itself
before the sbatch script's own commands run, so it can't be fixed from
inside the script -- a one-time `mkdir -p /hppfs/scratch/0D/go76fuz2/tmp`
on the login node would clear it -- but it's a local-vs-shared-tmp issue,
which wouldn't explain a *loopback* connection also hanging, and every
earlier 1-node job hit the same TMPDIR fallback without failing.

**Fix applied to both `finetune_vae_diag_2node.sbatch` and
`finetune_vae.sbatch`:** `export USE_LIBUV=0`, forcing the older TCPStore
backend. Cheap/safe regardless of whether this is really the cause: it's a
no-op for every already-working 1-node job (`num_machines=1` never
exercises the real multi-node TCPStore path this flag affects). The diag
script additionally sets `TORCH_DISTRIBUTED_DEBUG=DETAIL` and
`PYTHONUNBUFFERED=1` so that if `USE_LIBUV=0` alone doesn't fix it, the
*next* failure comes with far more to go on than a bare timeout.

**NOT YET RE-RUN.** This fix has not been tested against real hardware.
Launch with:
```
sbatch jobs/sng_pvc/finetune_vae_diag_2node.sbatch
```
If it still hangs at the same `_assign_worker_ranks`/`role_info` point even
with `USE_LIBUV=0`, the `TORCH_DISTRIBUTED_DEBUG=DETAIL` log should show
what the store is actually doing (or not doing) during the hang -- read
that before trying anything else. If it fails differently or earlier,
that's also useful signal (rules libuv out, points elsewhere).

## Eval parallelization fix, CONFIRMED WORKING (2026-08-06)

**Motivating question:** why does eval (675 sims) take almost as long per
epoch as train (3825 sims, ~5.7x more data), e.g. `finetune_vae_baseline_tail3`
(job 521395): train averaged 1220s/epoch, eval 853s/epoch -- eval is 41% of
total wall-clock despite the much smaller dataset.

**Root cause, found by reading `windinet/training/vae_trainer.py`:** eval was
never actually parallelized like train is.

1. The whole eval+log+viz+checkpoint block ran inside `if IS_MAIN_PROCESS:`
   -- only rank 0 ever called `_evaluate()`. On an 8-tile sng_pvc run, the
   other 7 ranks sat idle in `wait_for_everyone()` for the entire eval
   phase while rank 0 processed all 675 samples alone. Train, by contrast,
   is sharded 8-way (`train_loader = self._accelerator.prepare(train_loader)`,
   `train.py` around the optimizer setup) -- ~478 samples/rank, in parallel.
   Comparing the *per-active-rank* sample count is the right comparison:
   675 (eval, 1 rank) vs. 478 (train, per rank, x8 in parallel) -- eval was
   never the "smaller" job from a single rank's point of view.
2. `eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False)` had no
   `num_workers` argument -- always 0 (synchronous, no prefetch), silently
   ignoring `cfg.data.num_dataloader_workers` regardless of its value. (This
   is also why the `workers: 2 -> 4` cluster-default promotion above almost
   certainly didn't do what its own read-out claimed -- see the correction
   in "sng_pvc throughput diagnostic".)

(A third suspected cause, `_evaluate` missing `@torch.no_grad()`, was
**checked and is wrong** -- the decorator was already there; a first read
that started exactly on the `def` line missed it on the line above. Not a
real factor, noted here only so it isn't re-suspected later.)

**Fix**, `windinet/training/vae_trainer.py`:

- `eval_loader` is now built from a hand-strided per-rank shard
  (`Subset(eval_set, list(range(rank, len(eval_set), world_size)))`)
  instead of `accelerator.prepare()` -- deliberately not using `prepare()`,
  because its default `even_batches` padding repeats a few samples across
  ranks to equalize batch counts, which would double-count them in the
  metric average. The strided split has zero padding: every sample lands on
  exactly one rank, shard sizes differ by at most 1 (checked for 675/8:
  sizes 85,85,85,84,84,84,84,84, sums to 675 exactly). It also now gets
  `num_workers=cfg.data.num_dataloader_workers`, same as `train_loader`.
- The `_evaluate()` call moved outside the `IS_MAIN_PROCESS` gate -- every
  rank now calls it (over its own shard); the surrounding log/metrics/viz/
  checkpoint code stays `IS_MAIN_PROCESS`-only, using `val_metrics` computed
  above.
- `_evaluate()` all-reduces its per-rank sums (`total_loss`, `vrmse`,
  `rmse`, `h1`, `ssim`, `mlw`, `count`) with `dist.all_reduce(..., op=SUM)`
  before averaging, in float32 -- same manual-all-reduce-instead-of-DDP
  pattern `_sync_grads` already uses for gradients, and for the same reason
  (the VAE isn't DDP-wrapped, so nothing does this automatically). Guarded
  by `self._accelerator.num_processes > 1`, matching `_sync_grads`'s guard,
  so single-process runs (the overfit diagnostics, single-tile debugging)
  are unaffected.
- `_save_visualization` still reads from `eval_loader`, called on rank 0
  only -- it now sees rank 0's *shard* (every `world_size`-th sample
  starting at index 0) instead of the full eval set's first
  `vis_cfg.num_samples` samples. Index 0 is still included (rank 0's stride
  starts at 0), so the reconstruction panels won't be empty or wildly
  different, but the exact sample IDs visualized each epoch will shift
  slightly from pre-fix runs -- not a correctness issue, just don't expect
  panel sample IDs to match older runs' when comparing side by side.

**Expected effect:** eval should drop from ~one-rank-serial-675-samples to
~one-rank-parallel-84/85-samples-with-prefetch -- directionally a large cut
to the eval share of wall-clock (currently ~41% of total), though the exact
number needs a real run to confirm (rough pre-fix numbers, not a promise:
if eval scaled purely with shard size it'd be ~8x fewer samples per rank
plus whatever the worker prefetch buys, but network-filesystem contention
under concurrent multi-rank reads is untested and could eat into that).

**CONFIRMED, job 522099 (`DIAG_TAG=evalfix_confirm`, 2 epochs,
`finetune_vae_baseline.yaml`):**

| epoch | train= | eval= (pre-fix was ~800-1050s) |
|---|---|---|
| 1 | 895.9s | **38.1s** |
| 2 | 868.1s | **38.0s** |

**~22-27x faster** than the pre-fix per-epoch eval times seen throughout
this ledger (e.g. `finetune_vae_baseline_tail3`'s 793-1048s/epoch) -- far
beyond the "directionally large, exact number TBD" estimate above. Eval
went from ~41% of total wall-clock to a rounding error. Correctness check:
val_vrmse landed at 0.297 (epoch 1, lr still mid-warmup) -> 0.136 (epoch 2,
schedule already compressed to the floor for a 2-epoch diagnostic run) --
sane, in-range numbers, no NaN/blowup, consistent with the all-reduce
producing a real average rather than a corrupted one. No crash, no
DSS-locking symptom despite 8 ranks now reading `train.h5` concurrently
during eval too.

**File-locking question (Exp 4) ALSO ANSWERED, job 522100
(`DIAG_TAG=filelock_true DIAG_FILE_LOCKING=TRUE`):** finished clean, no
"Unable to synchronously open object" crash. Timing statistically
identical to 522099 (train 897.1s/869.8s, eval 37.9s/37.9s -- within noise
of the `FALSE` run). **Conclusion: `HDF5_USE_FILE_LOCKING=FALSE` is not
(or no longer) load-bearing for eval's newly-concurrent reads** -- but
since `FALSE` costs nothing and already works, there's no reason to flip
the default; this just confirms the eval fix didn't quietly reintroduce
the crash risk `FALSE` was set to avoid on the train side.

Both jobs used `finetune_vae_baseline.yaml` (`jobs/sng_pvc/finetune_vae_diag.sbatch`'s
default `BASE_CONFIG`), kept separate from the model-architecture sweeps
(head-vs-tail, encoder-LR) so infra effects aren't confounded with which
encoder modules are unfrozen -- launched together:

```
DIAG_TAG=evalfix_confirm sbatch jobs/sng_pvc/finetune_vae_diag.sbatch
DIAG_TAG=filelock_true DIAG_FILE_LOCKING=TRUE sbatch jobs/sng_pvc/finetune_vae_diag.sbatch
```

## Where things live

| | |
|---|---|
| Rationale, hypotheses, verdicts | **this file** |
| Cluster job launchers | `jobs/lundquist/*.sbatch`, `jobs/sng_pvc/*.sbatch` (submit from repo root) |
| Portable entry points | `scripts/*.py` |
| Slurm logs | `log_finetuning_vae/lundquist/<jobname>_<jobid>.log`, `log_finetuning_vae/sng_pvc/<jobid>-<jobname>.{out,err}` |
| job -> config -> output_dir map | `log_finetuning_vae/{lundquist,sng_pvc}/INDEX.tsv` (appended by the sbatch scripts) |
| Per-run metrics, panels, resolved config | `finetune_vae_outputs/lundquist/<run>/{metrics,visualizations,training_config.yaml}` -- **tracked in git** since `f88eb09` |
| Checkpoints | `finetune_vae_outputs/lundquist/<run>/checkpoints/vae_shockwave_best.safetensors` -- **gitignored**, this machine is the only copy |
| Retired runs | git history, see Artifact retention under the Ledger |
| Closed configs/outputs (2026-08-06 cleanup, extended 2026-08-08, 2026-08-12) | `configs/finetune_vae/archive/done/` + `finetune_vae_outputs/{lundquist,sng_pvc}/archive/done/<run>/` for finished, citable results (capacity diagnostics, full head-vs-tail sweep, encoder-LR sweep, channel-order sweep v1+v2, copy-init, seed noise floor sweep, decoder/adapter LR sweep, log-density, RMSE-only, 18-epoch confirmation, loss-function retest batch); `.../archive/known-bad/` for runs whose numbers are flagged uninterpretable (the 8-sim diagnostic tier of the head-vs-tail sweep, capacity attempt 2, GradNorm) -- see each folder's `README.md` before reusing anything inside |

**Directory rename (2026-08-12):** the two per-cluster output trees moved
from top-level `finetune_vae_outputs_lundquist/` / `finetune_vae_outputs_
sng_pvc/` to `finetune_vae_outputs/lundquist/` / `finetune_vae_outputs/
sng_pvc/` (one shared parent, cluster as a subdirectory) -- pure
`git mv`, no content changes. `windinet/cluster_config.py`'s
`output_root` entries and the two hardcoded `LOCAL_DIR` mirror targets in
`jobs/sng_pvc/finetune_vae.sbatch`/`finetune_vae_debug.sbatch` were
updated to match. Historical citations elsewhere in this file that point
into a specific git commit (e.g. `git show f88eb09:finetune_vae_outputs_
lundquist/...`) were deliberately left unchanged -- that commit's tree
still has the pre-rename layout, so rewriting the citation would make it
wrong, not right.

