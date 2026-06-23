"""
Checkpoint management for WinDiNet.

Downloads pretrained checkpoints from HuggingFace Hub on first use
and caches them locally. All scripts that need checkpoints should
call ``ensure_checkpoint()`` to resolve paths.
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

HF_REPO = "rabischof/windinet"

# Mapping from logical name to filename on HuggingFace Hub
CHECKPOINT_FILES = {
    "dit": "dit.safetensors",
    "scalar_embedding": "scalar_embedding.safetensors",
    "vae_decoder": "vae_decoder.safetensors",
}


def ensure_checkpoint(name_or_path: str) -> str:
    """Resolve a checkpoint path, downloading from HuggingFace if needed.

    Args:
        name_or_path: Either a logical name ("dit", "scalar_embedding",
            "vae_decoder") or a filesystem path. If the path exists on
            disk, it is returned as-is. If it matches a known logical name,
            the file is downloaded from HuggingFace Hub (cached automatically
            by huggingface_hub). Otherwise, raises FileNotFoundError.

    Returns:
        Absolute path to the checkpoint file.
    """
    # Already exists on disk — return as-is
    p = Path(name_or_path)
    if p.exists():
        return str(p.resolve())

    # Check if it's a known logical name
    if name_or_path in CHECKPOINT_FILES:
        filename = CHECKPOINT_FILES[name_or_path]
        return hf_hub_download(repo_id=HF_REPO, filename=filename)

    # Check if the basename matches a known file (e.g. "dit.safetensors")
    basename = p.name
    if basename in CHECKPOINT_FILES.values():
        return hf_hub_download(repo_id=HF_REPO, filename=basename)

    raise FileNotFoundError(
        f"Checkpoint not found: {name_or_path}\n"
        f"Provide a valid path or one of: {', '.join(CHECKPOINT_FILES.keys())}"
    )
