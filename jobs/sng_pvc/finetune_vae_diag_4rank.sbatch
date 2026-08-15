#!/bin/bash
#SBATCH --job-name=wh_diag_vae_4rank
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=general           # NOT "test" (30 min cap) -- see finetune_vae_diag.sbatch's header for why; this run already strips ckpt/viz I/O but train+eval alone is still an unknown, don't risk the cap
#SBATCH --account=pn82ku
#SBATCH -o log_finetuning_vae/sng_pvc/%j-%x.out
#SBATCH -e log_finetuning_vae/sng_pvc/%j-%x.err

# H2 (communication-bound) probe for the sng_pvc slowdown. RUN THIS LAST --
# only after jobs/sng_pvc/finetune_vae_diag.sbatch's Exp 0/2 have ruled out
# (or subtracted out) per-epoch checkpoint/viz I/O, otherwise that I/O will
# dominate both runs' "total=" and hide any communication difference.
#
# THE QUESTION: baseline runs 8 ranks x accum 4 (4 local micro-batches
# between allreduces). lundquist's 2-GPU baseline runs 2 ranks x accum 16
# (16 local micro-batches between allreduces) -- 4x more compute hidden
# behind each sync. If gradient allreduce over the 552.9M-param decoder is
# slow on this node's interconnect, sng_pvc pays for it 4x more often per
# unit of local compute than lundquist does. This run drops to 4 ranks
# (accum doubles to 8 automatically via patch_config_for_cluster's
# effective_batch=32 math) to see if halving the rank count measurably
# changes "train=" in the per-epoch timing line vae_trainer.py now prints.
#
# CAVEAT, VERIFY BEFORE TRUSTING THE RESULT: the baseline script sets
# ZE_FLAT_DEVICE_HIERARCHY=FLAT, which exposes each of the 4 physical Max
# 1550 GPUs' 2 tiles as 8 independent Level-Zero devices (matches the 8
# ranks). Here NUM_XPUS is dropped to 4, but for that to mean "1 rank = 1
# full physical GPU (both tiles via Intel implicit scaling)" rather than
# "1 rank = 1 tile, other 4 tiles idle", the device hierarchy needs to be
# COMPOSITE, not FLAT -- untested on this node. Check the job's "xpu-smi
# discovery" output and, if available, `xpu-smi topology` or equivalent to
# confirm 4 ranks actually map to 4 distinct physical GPUs (PCI BDF
# 3a/4b/ca/da:00.0 per the Exp 0 log) with both tiles active, not 4 ranks
# piled onto fewer physical GPUs. If COMPOSITE doesn't give clean 1
# rank/GPU behavior, the fallback is ZE_AFFINITY_MASK per rank instead of
# changing the hierarchy -- not attempted here since it needs a working
# per-rank launcher wrapper, out of scope for this first probe.
#
# Weights are throwaway (output_suffix "_diag_4rank"), not mirrored to git.

set -eo pipefail

module load micromamba/2.1.1
export PATH="$MICROMAMBA_BASE/bin:$PATH"
eval "$(micromamba shell hook -s bash)"
micromamba activate windinet

cd /dss/dsshome1/0D/go76fuz2/windinet

echo "START TIME: $(date)"
echo "job=${SLURM_JOB_ID} node=$(hostname)"
xpu-smi discovery || true

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=$(python - <<'PY'
import socket
s = socket.socket()
s.bind(('', 0))
print(s.getsockname()[1])
s.close()
PY
)
echo "MASTER_ADDR: $MASTER_ADDR  MASTER_PORT: $MASTER_PORT"

export NUM_XPUS=4
export CCL_ATL_TRANSPORT=ofi
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE   # see header caveat -- verify this actually yields 1 rank/physical GPU
export CCL_ALLREDUCE=ring

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export HDF5_USE_FILE_LOCKING=FALSE

BASE_CONFIG=${1:-configs/finetune_vae/finetune_vae_baseline.yaml}
RUN_CONFIG=$(mktemp /tmp/finetune_vae_sng_pvc_diag4_XXXX.yaml)
trap 'rm -f "${RUN_CONFIG}"' EXIT

python - "${BASE_CONFIG}" "${RUN_CONFIG}" "$((NUM_XPUS * SLURM_JOB_NUM_NODES))" <<'PY'
import sys

import yaml

from windinet.cluster_config import patch_config_for_cluster

src, dst, num_processes = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = yaml.safe_load(open(src))
patch_config_for_cluster(cfg, cluster="sng_pvc", num_processes=num_processes, output_suffix="_diag_4rank")

# Same throwaway-diagnostic trim as finetune_vae_diag.sbatch's Exp 2, so this
# run's "train=" isn't polluted by per-epoch checkpoint/viz I/O either.
# 2 epochs: epoch1=warmup, epoch2=clean reading -- see finetune_vae_diag.sbatch's header.
cfg["optimization"]["epochs"] = 2
cfg["checkpoints"]["interval"] = 1
cfg["checkpoints"]["save_last_state"] = False
cfg["visualization"]["enabled"] = False

yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(
    f"{num_processes} XPU ranks (COMPOSITE) x accum {cfg['optimization']['gradient_accumulation_steps']} "
    f"-> effective batch 32; output_dir {cfg['output_dir']}"
)
PY

OUT_DIR=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "${RUN_CONFIG}")
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" \
    "${BASE_CONFIG}" "${OUT_DIR}" >> log_finetuning_vae/sng_pvc/INDEX.tsv

srun --mpi=none accelerate launch \
    --num_processes $((NUM_XPUS * SLURM_JOB_NUM_NODES)) \
    --num_machines $SLURM_JOB_NUM_NODES \
    --machine_rank $SLURM_PROCID \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    --mixed_precision bf16 \
    scripts/finetune_vae.py "${RUN_CONFIG}"

echo "END TIME: $(date)"
echo "Diagnostic run -- weights/state left on \$SCRATCH under ${OUT_DIR}, NOT mirrored to git (throwaway)."
