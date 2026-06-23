"""
Wind field dataset for VAE decoder finetuning.

Loads raw CFD simulation data (u, v velocity fields + building mask) and
constructs conditioning videos compatible with the LTX-Video VAE.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class WindFieldDataset(Dataset):
    """Load wind field simulations from disk.

    Expected directory structure::

        root/<sample_id>/
            fields.npz     # keys: u_fields [T,H,W], v_fields [T,H,W], bldg_mask [H,W]
            meta.json      # must contain "wind_speed_mps"

    Args:
        root: path to dataset split (e.g. /data/wind_dataset/train).
        ids: optional list of sample IDs to use. If None, uses all subdirectories.
        wind_norm: velocity normalisation constant (divides u, v).
    """

    def __init__(self, root: str | Path, ids: list[str] | None = None, wind_norm: float = 30.0):
        self.root = Path(root)
        self.ids = sorted([p.name for p in self.root.iterdir() if p.is_dir()]) if ids is None else list(ids)
        self.wind_norm = wind_norm

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict[str, Tensor | str]:
        sid = self.ids[idx]
        d = self.root / sid

        with np.load(d / "fields.npz") as f:
            u = torch.from_numpy(f["u_fields"]).float() / self.wind_norm  # [T, H, W]
            v = torch.from_numpy(f["v_fields"]).float() / self.wind_norm
            b = 1 - torch.from_numpy(f["bldg_mask"]).unsqueeze(0).float()  # [1, H, W], 0=building, 1=fluid

        meta = json.loads((d / "meta.json").read_text())
        if isinstance(meta, list):
            meta = meta[0]
        wind_speed = torch.tensor(float(meta["wind_speed_mps"]), dtype=torch.float32) / self.wind_norm

        return {"u": u, "v": v, "b": b, "wind_speed": wind_speed, "id": sid}


def build_conditioning_video(sample: dict[str, Tensor], device: torch.device) -> Tensor:
    """Build a 3-channel video [B, 3, F, H, W] with conditioning frame at t=0.

    Channel layout: [u, v, b] in [-1, 1].

    The conditioning frame (t=0) encodes the inlet boundary condition:
      - u channel: inlet wind speed broadcast over fluid pixels
      - v channel: zero (no crosswind)
      - b channel: building footprint (-1 = building, +1 = fluid)

    Frames t=1.. contain the simulation data. The total frame count is
    padded to 8n+1 as required by the LTX-Video VAE.
    """
    u, v, b = sample["u"], sample["v"], sample["b"]
    wind_speed = sample["wind_speed"]

    # Ensure batch dimension
    if u.ndim == 3:
        u = u.unsqueeze(0)
    if v.ndim == 3:
        v = v.unsqueeze(0)
    if b.ndim == 3:
        b = b.unsqueeze(0)
    if wind_speed.ndim == 0:
        wind_speed = wind_speed.unsqueeze(0)

    B, T, H, W = u.shape
    b0 = b[:, 0]  # [B, H, W]

    # Conditioning frame
    u0 = wind_speed.view(B, 1, 1, 1).expand(B, 1, H, W) * b0.unsqueeze(1)  # inlet speed in fluid
    v0 = torch.zeros((B, 1, H, W), device=u.device, dtype=u.dtype)
    b0_frame = b0.unsqueeze(1)  # [B, 1, H, W]

    # Prepend conditioning frame
    u_full = torch.cat([u0, u], dim=1)  # [B, 1+T, H, W]
    v_full = torch.cat([v0, v], dim=1)
    b_full = torch.cat([b0_frame, b0_frame.expand(-1, T, -1, -1)], dim=1)

    # Normalise to [-1, 1]
    u_full = u_full.clamp(-1, 1)
    v_full = v_full.clamp(-1, 1)
    b_full = (b_full * 2 - 1).clamp(-1, 1)  # {0,1} -> {-1,+1}

    x = torch.stack([u_full, v_full, b_full], dim=1)  # [B, 3, F, H, W]
    x = pad_frames_8n1(x)
    return x.to(device=device, dtype=torch.float32).contiguous()


def pad_frames_8n1(x: Tensor) -> Tensor:
    """Pad temporal dimension so F = 8n + 1 (required by LTX-Video VAE).

    Pads by repeating the last frame.

    Args:
        x: [B, C, F, H, W] tensor.

    Returns:
        Padded tensor with F' = 8n + 1 >= F.
    """
    orig_F = x.shape[2]
    target = ((orig_F - 1 + 7) // 8) * 8 + 1
    if target == orig_F:
        return x
    pad_frames = target - orig_F
    last = x[:, :, -1:]
    return torch.cat([x, last.expand(-1, -1, pad_frames, -1, -1)], dim=2)
