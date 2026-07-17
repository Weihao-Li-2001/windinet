"""
ShockWave dataset for ShockWaveNet.

Loads ShockWave CFD simulations from HDF5 file.

Expected structure:

train.h5

    <sample_id>/
        density       [T,1,H,W]
        momentum_x    [T,1,H,W]
        momentum_y    [T,1,H,W]
        pressure      [T,1,H,W]

Returns physical fields for LTX preprocessing.
"""

from pathlib import Path

import h5py
import torch
import torch
from torch import Tensor
from torch.utils.data import Dataset


class ShockWaveDataset(Dataset):
    """
    Load ShockWave CFD simulations from HDF5.

    Each sample corresponds to one simulation.

    Returns:
        density:
            [T,1,H,W]

        momentum_x:
            [T,1,H,W]

        momentum_y:
            [T,1,H,W]

        pressure:
            [T,1,H,W]

        gamma:
            scalar condition

        id:
            sample name
    """

    def __init__(
        self,
        h5_path: str | Path,
        num_sim_frames: int | None = None,
    ):
        self.h5_path = Path(h5_path)

        self.file = None

        with h5py.File(self.h5_path, "r") as f:
            self.ids = sorted(list(f.keys()))

        self.num_sim_frames = num_sim_frames


    def __len__(self):
        return len(self.ids)


    def _init_file(self):
        """
        Lazy open h5 file.
        Important for DataLoader workers.
        """
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")

    def __del__(self):
        if self.file is not None:
            self.file.close()

    def __getitem__(self, idx):

        self._init_file()

        sid = self.ids[idx]

        sample = self.file[sid]


        density = torch.from_numpy(
            sample["density"][:]
        ).float().squeeze(1)


        momentum_x = torch.from_numpy(
            sample["momentum_x"][:]
        ).float().squeeze(1)


        momentum_y = torch.from_numpy(
            sample["momentum_y"][:]
        ).float().squeeze(1)


        pressure = torch.from_numpy(
            sample["pressure"][:]
        ).float().squeeze(1)


        if self.num_sim_frames is not None:

            density = density[:self.num_sim_frames]

            momentum_x = momentum_x[:self.num_sim_frames]

            momentum_y = momentum_y[:self.num_sim_frames]

            pressure = pressure[:self.num_sim_frames]


        # parse gamma from name
        # example:
        # 0000_gamma1.2200000286

        gamma = float(
            sid.split("gamma")[1]
        )


        gamma = torch.tensor(
            gamma,
            dtype=torch.float32
        )


        return {
            "density": density,
            "momentum_x": momentum_x,
            "momentum_y": momentum_y,
            "pressure": pressure,
            "meta": {
                "gamma": gamma.item()
            },
            "id": sid,
        }
    
def build_shockwave_video(
    sample: dict[str, Tensor],
    device: torch.device,
) -> Tensor:
    """Build a 4-channel shockwave video [B, 4, F, H, W].

    Channel layout:
        0 - density
        1 - momentum_x
        2 - momentum_y
        3 - pressure

    Unlike WinDiNet, ShockWaveNet does not prepend an artificial
    conditioning frame. The first simulation frame is treated as
    the physical initial condition (IC).

    The temporal dimension is padded to 8n+1 as required by the
    LTX-Video VAE.
    """

    density = sample["density"]
    momentum_x = sample["momentum_x"]
    momentum_y = sample["momentum_y"]
    pressure = sample["pressure"]

    # Ensure batch dimension
    if density.ndim == 3:
        density = density.unsqueeze(0)

    if momentum_x.ndim == 3:
        momentum_x = momentum_x.unsqueeze(0)

    if momentum_y.ndim == 3:
        momentum_y = momentum_y.unsqueeze(0)

    if pressure.ndim == 3:
        pressure = pressure.unsqueeze(0)

    # density:    [B, T, H, W]
    # momentum_x: [B, T, H, W]
    # momentum_y: [B, T, H, W]
    # pressure:   [B, T, H, W]

    x = torch.stack(
        [
            density,
            momentum_x,
            momentum_y,
            pressure,
        ],
        dim=1,
    )  # [B, 4, F, H, W]

    x = pad_frames_8n1(x)

    return x.to(
        device=device,
        dtype=torch.float32,
    ).contiguous()
    
def extract_scalars(
    meta: dict,
    scalar_names: list[str] | None = None,
) -> dict[str, float]:
    """Extract scalar conditioning values from a bubble sample's metadata.

    Args:
        meta: metadata dictionary.
        scalar_names: names to extract. Defaults to ["gamma"].

    Returns:
        Dictionary mapping scalar names to float values.
    """
    if scalar_names is None:
        scalar_names = ["gamma"]

    scalars = {}

    for name in scalar_names:
        scalars[name] = float(meta[name])

    return scalars

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