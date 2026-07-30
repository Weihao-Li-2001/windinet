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

import json
import os
import random
import time
from pathlib import Path

import h5py
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

        if not self.h5_path.is_file():
            raise FileNotFoundError(f"Shockwave HDF5 file not found: {self.h5_path}")

        time.sleep(self._rank_stagger_delay())
        with self._open(self.h5_path) as f:
            self.ids = sorted(list(f.keys()))
            self._prewarm_ids = self._pick_prewarm_ids(f, self.ids)

            # DEBUG: prove which file is actually being read, not just what the
            # config claims. Rank0 only to avoid 8x duplicate spam.
            if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0:
                real_path = os.path.realpath(self.h5_path)
                print(f"[ShockWaveDataset DEBUG] h5_path        = {self.h5_path}", flush=True)
                print(f"[ShockWaveDataset DEBUG] realpath        = {real_path}", flush=True)
                print(f"[ShockWaveDataset DEBUG] st_size (bytes) = {os.stat(self.h5_path).st_size}", flush=True)
                print(f"[ShockWaveDataset DEBUG] first 5 keys    = {self.ids[:5]}", flush=True)
                print(f"[ShockWaveDataset DEBUG] external targets to prewarm = {len(self._prewarm_ids)}", flush=True)

        self.num_sim_frames = num_sim_frames

    @staticmethod
    def _pick_prewarm_ids(f: h5py.File, ids: list[str]) -> list[str]:
        """
        One sample id per distinct external-link target file.

        Each sample in train.h5 is an ``h5py.ExternalLink`` into a separate
        per-gamma shard file (e.g. ``train_subsets/train_gamma1.130_128.hdf5``).
        Resolving a link for the first time makes HDF5 open its target file;
        with several ranks shuffling through samples independently, those
        first-opens land at unpredictable, possibly simultaneous points deep
        into training instead of a controlled startup window. `_prewarm_external_targets`
        uses this list to force one resolution per distinct target up front,
        while ranks are still staggered, so that burst happens once and only
        once rather than scattered through the epoch.
        """
        seen_targets = set()
        prewarm_ids = []
        for sid in ids:
            link = f.get(sid, getlink=True)
            target = getattr(link, "filename", None)
            if target is None:
                # Not an external link (e.g. a self-contained h5 file) --
                # nothing to pre-resolve.
                continue
            if target not in seen_targets:
                seen_targets.add(target)
                prewarm_ids.append(sid)
        return prewarm_ids


    def __len__(self):
        return len(self.ids)


    @staticmethod
    def _rank_stagger_delay(spread: float = 0.3) -> float:
        """
        Per-rank delay to spread out each process's first file access.

        Every rank in a distributed run reaches its first HDF5 access at
        nearly the same wall-clock instant (dataset construction, then the
        first `__getitem__`), so without this, all ranks hit the network
        filesystem in the same burst regardless of the retry jitter in
        `_backoff_sleep` -- that only helps *after* a collision has already
        happened. Staggering by rank spreads the initial burst out
        preemptively. Falls back to 0 outside a distributed run.
        """
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        return rank * spread

    @staticmethod
    def _backoff_sleep(attempt: int) -> None:
        """
        Sleep with exponential backoff plus random jitter.

        All ranks in a distributed run hit the dataset near-simultaneously
        (e.g. all fetching the first batch at once), so a fixed backoff
        schedule with no jitter has every rank retry in lockstep -- they
        keep re-colliding on the network filesystem at the same instants
        instead of spreading out. The jitter desynchronizes them.
        """
        base = 0.5 * (2 ** attempt)
        time.sleep(base + random.uniform(0, base))

    @staticmethod
    def _open(path: Path, retries: int = 8) -> h5py.File:
        """
        Open the HDF5 file, retrying with backoff.

        train.h5 stores each sample as an h5py.ExternalLink into a separate
        per-gamma shard file (e.g. train_subsets/train_gamma1.130_128.hdf5).
        On this environment's h5py/HDF5 build, resolving that link depends on
        the process's current working directory rather than falling back to
        train.h5's own directory -- confirmed by the link resolving fine when
        opened with the process cwd'd into the same directory as train.h5,
        but failing (single process, no concurrency involved) when opened by
        absolute path from a different cwd, exactly the situation training
        runs in. Setting HDF5_EXT_PREFIX to this file's own directory makes
        link resolution explicit instead of relying on that cwd-dependent
        fallback. The retry-with-backoff below is a separate, unrelated
        mitigation for genuine transient network-filesystem hiccups on LRZ
        DSS and is kept as defense in depth.
        """
        os.environ["HDF5_EXT_PREFIX"] = str(path.parent)
        for attempt in range(retries):
            try:
                try:
                    return h5py.File(path, "r", locking=False)
                except TypeError:
                    # Installed h5py/libhdf5 too old to support the `locking` kwarg.
                    return h5py.File(path, "r")
            except OSError:
                if attempt == retries - 1:
                    raise
                ShockWaveDataset._backoff_sleep(attempt)
        raise AssertionError("unreachable")

    def _init_file(self):
        """
        Lazy open h5 file.
        Important for DataLoader workers.
        """
        if self.file is None:
            time.sleep(self._rank_stagger_delay())
            self.file = self._open(self.h5_path)

            # DEBUG: does a brand-new h5py.File opened right here, inside the
            # live (already accelerate/torch/XPU-initialized) process, also
            # fail to resolve the same external link that every standalone
            # `python3 -c` repro outside this process has resolved fine?
            # Distinguishes "something wrong with this Dataset's file handle"
            # from "something about this process's state at this point breaks
            # HDF5 external-link resolution regardless of which handle is used".
            if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0 and self._prewarm_ids:
                probe_id = self._prewarm_ids[0]
                try:
                    probe_file = h5py.File(self.h5_path, "r", locking=False)
                    _ = probe_file[probe_id]
                    print(f"[ShockWaveDataset DEBUG] fresh-handle probe OK for {probe_id}", flush=True)
                    probe_file.close()
                except Exception as e:
                    print(f"[ShockWaveDataset DEBUG] fresh-handle probe FAILED for {probe_id}: {e!r}", flush=True)

            for sid in self._prewarm_ids:
                self._get_group(sid)

    def __del__(self):
        if self.file is not None:
            self.file.close()

    def _get_group(self, sid: str, retries: int = 8):
        """
        Look up one simulation's group, retrying with a fresh file handle.

        Under multi-rank distributed training, `self.file[sid]` has been
        observed to transiently fail with "Unable to synchronously open
        object (can't open file)" even though `self.file` itself opened fine
        -- this never happens in a single-process run, so it looks like
        contention on the network filesystem rather than a real missing key.
        Retry with a freshly reopened handle rather than trusting the stale
        one. See `_backoff_sleep` for why the retries are jittered.
        """
        for attempt in range(retries):
            try:
                return self.file[sid]
            except KeyError:
                if attempt == retries - 1:
                    raise
                self._backoff_sleep(attempt)
                self.file.close()
                self.file = self._open(self.h5_path)
        raise AssertionError("unreachable")

    def __getitem__(self, idx):

        self._init_file()

        sid = self.ids[idx]

        sample = self._get_group(sid)


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
    channel_mean: list[float] | None = None,
    channel_std: list[float] | None = None,
    normalization_clip: float = 5.0,
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

    if channel_mean is not None and channel_std is not None:
        mean = x.new_tensor(channel_mean).view(1, 4, 1, 1, 1)
        std = x.new_tensor(channel_std).view(1, 4, 1, 1, 1)
        x = ((x - mean) / (std * normalization_clip)).clamp(-1.0, 1.0)

    x = pad_frames_8n1(x)

    return x.to(
        device=device,
        dtype=torch.float32,
    ).contiguous()
    
CHANNEL_NAMES = ["density", "momentum_x", "momentum_y", "pressure"]


def load_channel_normalization(source: str | Path) -> dict:
    """Read the channel normalization stats that a VAE / latent set was built with.

    Accepts either a ``normalization.json`` written by ``preprocess_dataset.py``
    or a VAE run's ``training_config.yaml``. Encoding, decoding and preprocessing
    must all agree on these numbers -- the VAE only ever saw fields normalized as
    ``(x - mean) / (std * clip)`` clamped to [-1, 1], so any mismatch silently
    produces off-distribution latents or wrongly scaled physical outputs.

    Returns a dict with ``channel_mean``, ``channel_std`` and ``normalization_clip``.
    """
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Normalization source not found: {path}")

    if path.suffix == ".json":
        payload = json.loads(path.read_text())
    else:
        import yaml

        # A dumped training_config.yaml carries python-tagged values (e.g. the
        # model-source enum). Ignore unknown tags rather than reaching for
        # unsafe_load, which would execute whatever the file names.
        class _TolerantLoader(yaml.SafeLoader):
            pass

        _TolerantLoader.add_multi_constructor("", lambda loader, suffix, node: None)
        payload = (yaml.load(path.read_text(), Loader=_TolerantLoader) or {}).get("data", {})

    mean, std = payload.get("channel_mean"), payload.get("channel_std")
    if not mean or not std:
        raise ValueError(f"{path} has no channel_mean / channel_std")
    return {
        "channel_mean": list(mean),
        "channel_std": list(std),
        "normalization_clip": float(payload.get("normalization_clip", 5.0)),
    }


def normalize_fields(
    x: Tensor,
    channel_mean: list[float],
    channel_std: list[float],
    normalization_clip: float,
) -> Tensor:
    """Normalize physical fields [B, C, ...] to [-1, 1]; inverse of denormalize_fields."""
    shape = (1, len(channel_mean)) + (1,) * (x.ndim - 2)
    mean = x.new_tensor(channel_mean).view(shape)
    scale = x.new_tensor(channel_std).view(shape) * normalization_clip
    return ((x - mean) / scale).clamp(-1.0, 1.0)


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
