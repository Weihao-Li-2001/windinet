"""Repo-local locations for downloaded model weights.

Set ``WINDINET_HF_CACHE`` to redirect the HuggingFace hub cache to a
cluster-local path (e.g. lundquist's in-repo ``pretrained/hub``, so a clone
is self-contained there and the same files resolve without per-machine
cache setup). **Opt-in per cluster, not a blanket default** -- left unset,
this module does nothing and ``huggingface_hub`` keeps using its own
default (``~/.cache/huggingface``). sng_pvc and lrz_ai deliberately do not
set it and must keep using their existing cache location: lrz_ai's `$HOME`
is DSS-quota'd and cannot absorb an ~8.4GB in-repo cache (job 5750198,
"Disk quota exceeded", 2026-08-15, from an earlier version of this module
that defaulted to `pretrained/hub` for every cluster).

The layout under a ``WINDINET_HF_CACHE``-pointed directory is a normal
HuggingFace hub cache (``models--<org>--<repo>/snapshots/<sha>/...``), so
``huggingface_hub`` can both read it offline and populate it with new
downloads.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: None unless WINDINET_HF_CACHE is set -- see module docstring. None means
#: "leave huggingface_hub's own default alone", not "use pretrained/hub".
HF_CACHE_DIR = Path(os.environ["WINDINET_HF_CACHE"]) if os.environ.get("WINDINET_HF_CACHE") else None


def use_repo_hf_cache() -> Path | None:
    """Redirect huggingface_hub to :data:`HF_CACHE_DIR` for this process and children,
    if ``WINDINET_HF_CACHE`` is set. No-op otherwise (returns ``None``).

    Call sites pass ``cache_dir=`` explicitly, so this only covers the paths we
    do not control (diffusers internals, subprocesses). It runs on ``import
    windinet``, which is early enough as long as ``windinet`` is imported before
    ``huggingface_hub`` reads its constants.
    """
    if HF_CACHE_DIR is None:
        return None
    cache = str(HF_CACHE_DIR)
    os.environ["HF_HUB_CACHE"] = cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache  # legacy alias, still read by older deps
    return HF_CACHE_DIR
