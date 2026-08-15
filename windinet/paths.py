"""Repo-local locations for downloaded model weights.

The LTX-Video base weights live inside the checkout (``pretrained/hub``) instead
of the user's ``~/.cache/huggingface``: a clone is then self-contained, and the
same files resolve on every cluster without per-machine cache setup.

The layout under ``pretrained/hub`` is a normal HuggingFace hub cache
(``models--<org>--<repo>/snapshots/<sha>/...``), so ``huggingface_hub`` can both
read it offline and populate it with new downloads.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: HuggingFace hub cache shipped with the repo. ``WINDINET_HF_CACHE`` overrides it
#: (e.g. to a shared scratch copy on a cluster).
HF_CACHE_DIR = Path(os.environ.get("WINDINET_HF_CACHE") or REPO_ROOT / "pretrained" / "hub")


def use_repo_hf_cache() -> Path:
    """Redirect huggingface_hub to :data:`HF_CACHE_DIR` for this process and children.

    Call sites pass ``cache_dir=`` explicitly, so this only covers the paths we
    do not control (diffusers internals, subprocesses). It runs on ``import
    windinet``, which is early enough as long as ``windinet`` is imported before
    ``huggingface_hub`` reads its constants.
    """
    cache = str(HF_CACHE_DIR)
    os.environ["HF_HUB_CACHE"] = cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache  # legacy alias, still read by older deps
    return HF_CACHE_DIR
