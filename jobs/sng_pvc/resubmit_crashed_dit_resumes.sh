#!/bin/bash
# One-off resubmission of the sng_pvc DiT resume arms that SIGKILL'd shortly
# after a mid-training checkpoint save (jobs 535059/535060/535061/535062,
# four different nodes -- host-RAM OOM from the self._resume_state leak
# fixed in ed536b2, not a bad node). Each *_resume.yaml's
# model.load_checkpoint points at the run's checkpoints/ directory, and
# LtxvTrainer._find_checkpoint auto-picks the max surviving step -- so
# resubmitting the same resume config just continues from wherever the
# crash left off, no config edits needed.
#
# UPDATE 2026-09-09: 535060 (cosine_kl1e7_resume) was originally left out of
# this script because as of 2026-09-08 it was still PD (queued) rather than
# crashed. It has since started, run to step ~2040, and SIGKILL'd the same
# way as the other three (same host-RAM leak, pre-ed536b2) -- adding it back
# in below.
#
# NOT included: ep20_baseline_std0p5 (job 535064) crashed differently --
# bare SIGSEGV at step 1930, no Python traceback, not the resume-state leak
# -- ed536b2 is unlikely to fix it. Needs separate investigation before
# resubmitting.
#
# Run from the repo root on sng_pvc (/dss/dsshome1/0D/go76fuz2/windinet),
# after `git pull` picks up ed536b2 (the resume-state leak fix) -- pure
# Python change, no env/reinstall needed.
set -euo pipefail

sbatch jobs/sng_pvc/train_dit.sbatch \
    finetune_vae_whole_structure_baseline_ep20_256res_cosine_kl1e6 \
    configs/dit/train_dit_sng_pvc_cosine_kl1e6_resume.yaml

sbatch jobs/sng_pvc/train_dit.sbatch \
    finetune_vae_whole_structure_baseline_ep20_256res_cosine_kl1e7 \
    configs/dit/train_dit_sng_pvc_cosine_kl1e7_resume.yaml

sbatch jobs/sng_pvc/train_dit.sbatch \
    finetune_vae_whole_structure_baseline_ep20_256res \
    configs/dit/train_dit_sng_pvc_ep20_baseline_lr3e5_resume.yaml

sbatch jobs/sng_pvc/train_dit.sbatch \
    finetune_vae_whole_structure_baseline_ep20_256res \
    configs/dit/train_dit_sng_pvc_ep20_baseline_lr3e6_resume.yaml
