# checked and changed

"""
VAE physics adapter for wind field encoding/decoding.

Wraps the LTX-Video VAE with lightweight adapters that transform between
n-channel wind fields (u, v, building mask) and 3-channel RGB space.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from windinet.utils import get_default_device


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"unknown activation: {name}")


def _strict_load(module: nn.Module, sd: dict[str, torch.Tensor], what: str) -> None:
    try:
        module.load_state_dict(sd, strict=True)
    except Exception as e:
        raise RuntimeError(f"strict load failed for {what}: {e}") from e


class InAdapter(nn.Module):
    """Maps n input channels to 3 RGB channels for the VAE encoder.

    Two modes:
      * default (``identity_init=False``): ``tanh(proj2(act(proj1(x))))``.
      * ``identity_init=True``: passthrough of the first 3 channels plus a
        residual whose last layer is zero-initialized, so at init the adapter
        equals the identity (``x[:, :3]``) instead of tanh's ~0 dead zone.
        This lets real spatial structure reach the encoder from step 0.
    """

    def __init__(self, n: int, k: int, activation: str, identity_init: bool = False):
        super().__init__()
        self.n = n
        self.k = k
        self.identity_init = bool(identity_init)
        if k > 0:
            self.proj1 = nn.Conv3d(n, k, kernel_size=1)
            self.act = _get_activation(activation)
            self.proj2 = nn.Conv3d(k, 3, kernel_size=1)
            if self.identity_init:
                nn.init.zeros_(self.proj2.weight)
                nn.init.zeros_(self.proj2.bias)

    def forward(self, x_n: torch.Tensor) -> torch.Tensor:
        if self.k == 0:
            return x_n
        y = self.proj2(self.act(self.proj1(x_n)))
        if self.identity_init:
            # Identity + residual (zero at init), clamped to the encoder's [-1, 1].
            return (x_n[:, :3] + y).clamp(-1.0, 1.0)
        return torch.tanh(y) # * torch.pi


class OutAdapter(nn.Module):
    """Maps 3 RGB channels from VAE decoder to n output channels.

    With ``identity_init=True`` the first 3 output channels start as the
    decoder's RGB output and any extra channels start at 0, plus a zero-init
    residual -- avoiding tanh's dead zone. ``AdaptedVAE.decode`` clamps the
    result to [-1, 1], so no output tanh is needed in this mode.
    """

    def __init__(self, n: int, k: int, activation: str, identity_init: bool = False):
        super().__init__()
        self.n = n
        self.k = k
        self.identity_init = bool(identity_init)
        if k > 0:
            self.proj1 = nn.Conv3d(3, k, kernel_size=1)
            self.act = _get_activation(activation)
            self.proj2 = nn.Conv3d(k, n, kernel_size=1)
            if self.identity_init:
                nn.init.zeros_(self.proj2.weight)
                nn.init.zeros_(self.proj2.bias)

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        if self.k == 0:
            return x_rgb
        y = self.proj2(self.act(self.proj1(x_rgb)))
        if self.identity_init:
            if self.n >= 3:
                b, _, f, h, w = x_rgb.shape
                passthrough = torch.cat(
                    [x_rgb, x_rgb.new_zeros(b, self.n - 3, f, h, w)], dim=1
                )
            else:
                passthrough = x_rgb[:, : self.n]
            return passthrough + y
        return torch.tanh(y) #  * torch.pi


class AdaptedVAE(nn.Module):
    """
    Wraps an LTX-Video VAE with input/output adapters for wind field data.

    All channels (u, v, b) are in [-1, 1]:
      - u, v: wind velocity normalized by wind_norm
      - b: footprint mask where -1 = building, +1 = fluid
    """

    DEFAULT_INFERENCE_TEMB: float = 0.025

    def __init__(self, vae, in_adapter, out_adapter, *, channels, k, default_temb=0.025):
        super().__init__()
        self.vae = vae
        self.in_adapter = in_adapter
        self.out_adapter = out_adapter
        self.channels = list(channels)
        self.k = int(k)
        self.default_temb = float(default_temb)

    @property
    def config(self):
        return self.vae.config

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.vae, name)

    def _as_bcfhw(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected 5d tensor, got {tuple(x.shape)}")
        n = int(getattr(self.in_adapter, "n", -1))
        if x.shape[1] in (3, n):
            return x.contiguous()
        if x.shape[2] in (3, n):
            return x.permute(0, 2, 1, 3, 4).contiguous()
        raise ValueError(f"cannot infer channel dimension for shape {tuple(x.shape)}")

    def encode(self, x: torch.Tensor, *args, **kwargs):
        x = self._as_bcfhw(x)
        x_rgb = self.in_adapter(x)
        return self.vae.encode(x_rgb, *args, **kwargs)

    def decode(self, *args, **kwargs):
        out = self.vae.decode(*args, **kwargs)

        def _postprocess(sample):
            return self.out_adapter(sample).clamp(-1.0, 1.0)

        if isinstance(out, (tuple, list)):
            if len(out) == 0 or not torch.is_tensor(out[0]):
                return out
            return (_postprocess(out[0]), *out[1:])

        if hasattr(out, "sample") and torch.is_tensor(out.sample):
            out.sample = _postprocess(out.sample)

        return out


def _clone_conv3d(old: nn.Conv3d, in_channels: int, out_channels: int) -> nn.Conv3d:
    """A new nn.Conv3d with the same hyperparameters but different channel counts."""
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    ).to(device=old.weight.device, dtype=old.weight.dtype)


@torch.no_grad()
def _reinit_scaled(new: nn.Conv3d, old: nn.Conv3d) -> None:
    """Discard the pretrained weights and keep the fresh init, rescaled to match.

    ``_clone_conv3d`` already returns a Conv3d carrying PyTorch's default
    Kaiming-uniform init, so "reinitialize" just means not overwriting it. The
    one thing that cannot be left to the default is the *scale*: a conv output
    has variance ~ fan_in * var(w) * var(x), and inflating 3 -> 4 channels grows
    fan_in from 3*block to 4*block. Left alone, the fresh layer would hand the
    frozen trunk activations of a different magnitude than the ones it was
    trained on, on top of the intended change of direction.

    So the fresh weight is rescaled to ``std(old) * sqrt(old_fan_in/new_fan_in)``,
    which reproduces the pretrained layer's *output* variance while keeping the
    randomized structure. This assumes the input channels are on comparable
    scales -- true here, since every channel is normalized by its own mean/std
    before reaching the VAE.

    The bias is left at its fresh init too: the point of this mode is that
    nothing pretrained survives in this layer.
    """
    old_fan_in = old.in_channels * math.prod(old.kernel_size)
    new_fan_in = new.in_channels * math.prod(new.kernel_size)
    target = old.weight.detach().float().std() * math.sqrt(old_fan_in / new_fan_in)
    current = new.weight.detach().float().std()
    if float(current) > 0.0:
        new.weight.mul_(target / current)


@torch.no_grad()
def inflate_vae_io_channels(vae, n: int = 4, init: str = "zeros"):
    """Make the LTX VAE natively read/write ``n`` channels instead of 3.

    The encoder patchifies (C x patch^2) before ``conv_in`` and the decoder
    unpatchifies after ``conv_out``; both operations infer the channel count
    dynamically from the tensor, so only the two projection convs need to grow:

      * ``encoder.conv_in``:  (3*block -> C0)  ->  (n*block -> C0)
      * ``decoder.conv_out``: (Cd -> 3*block)  ->  (Cd -> n*block)

    where ``block = patch_size**2 * patch_size_t`` (16 here). Channels are laid
    out blocked-by-channel (verified): original channel c occupies slots
    ``[c*block, c*block+block)``.

    ``init`` controls what the grown projections start from:

      * ``'zeros'`` (default) -- keep all pretrained slots, zero the new channel.
        Preserves the pretrained forward pass exactly and learns the new channel
        from clean gradients.
      * ``'mean'`` -- keep all pretrained slots, seed the new channel with the
        mean of the originals at the same patch position (I3D-style).
      * ``'random'`` -- **discard the pretrained projections entirely** and
        reinitialize both layers from scratch, rescaled to preserve the output
        variance (see :func:`_reinit_scaled`). The premise is that LTXV's
        patchify/unpatchify filters are tuned to natural-image RGB statistics
        and are the wrong basis for CFD fields, so relearning them beats
        adapting them. Note the asymmetric risk: ``decoder.conv_out`` is part of
        the fully-trainable decoder and is relearned either way, but
        ``encoder.conv_in`` is the *only* trainable module on the encoder side
        (221,312 params) feeding a frozen 694M trunk -- randomizing it means the
        trunk sees off-distribution input at step 0 and has no way to adapt.
        Expect a much worse epoch 1 than 'zeros'/'mean'; the bet is on the
        endpoint, not the start.
    """
    if init not in ("zeros", "mean", "random"):
        raise ValueError(f"init must be 'zeros', 'mean' or 'random', got {init!r}")
    n_orig = int(vae.config.in_channels)
    if n < n_orig:
        raise ValueError(f"n={n} must be >= original in_channels={n_orig}")

    def _fill_new_blocks(w_new: torch.Tensor, block: int, axis: int) -> None:
        # w_new indexed on `axis` as [c0_block, c1_block, ... c_{n-1}_block].
        for c_new in range(n_orig, n):
            for p in range(block):
                dst = c_new * block + p
                if init == "zeros":
                    continue  # already zeroed
                src = [c * block + p for c in range(n_orig)]
                if axis == 1:
                    w_new[:, dst] = w_new[:, src].mean(dim=1)
                else:
                    w_new[dst] = w_new[src].mean(dim=0)

    # --- encoder.conv_in: inflate INPUT channels ---
    ci = vae.encoder.conv_in
    old = ci.conv
    block = old.in_channels // n_orig
    new = _clone_conv3d(old, in_channels=n * block, out_channels=old.out_channels)
    if init == "random":
        _reinit_scaled(new, old)
    else:
        new.weight.zero_()
        new.weight[:, : old.in_channels] = old.weight
        _fill_new_blocks(new.weight, block, axis=1)
        if old.bias is not None:  # bias is per-output-channel -> unchanged
            new.bias.copy_(old.bias)
    ci.conv = new
    ci.in_channels = n * block
    vae.encoder.in_channels = n * block

    # --- decoder.conv_out: inflate OUTPUT channels ---
    co = vae.decoder.conv_out
    old = co.conv
    block = old.out_channels // n_orig
    new = _clone_conv3d(old, in_channels=old.in_channels, out_channels=n * block)
    if init == "random":
        # fan_in is unchanged here (only the output width grows), so the rescale
        # is a no-op in expectation; call it anyway so the two projections are
        # initialized by one code path.
        _reinit_scaled(new, old)
    else:
        new.weight.zero_()
        new.weight[: old.out_channels] = old.weight
        if old.bias is not None:
            new.bias.zero_()
            new.bias[: old.out_channels] = old.bias
        _fill_new_blocks(new.weight, block, axis=0)
        if old.bias is not None and init == "mean":
            for p in range(block):
                for c_new in range(n_orig, n):
                    new.bias[c_new * block + p] = new.bias[[c * block + p for c in range(n_orig)]].mean()
    co.conv = new
    co.out_channels = n * block
    vae.decoder.out_channels = n * block

    try:
        vae.register_to_config(in_channels=n, out_channels=n)
    except Exception:
        pass
    return vae


@torch.no_grad()
def load_inflated_vae_checkpoint(
    vae,
    ckpt_path: str | Path,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    verbose: bool = True,
) -> dict[str, str]:
    """Load a finetuned inflate-mode checkpoint into an already-inflated VAE.

    The checkpoint is the ``ltx-inflated-io-v1`` safetensors written by
    ``VaeTrainer._save_checkpoint``: it holds the finetuned ``decoder.*`` and the
    grown ``encoder_conv_in.*`` weights. ``vae`` must already have been passed
    through :func:`inflate_vae_io_channels` so the conv shapes match. Returns the
    checkpoint metadata dict.
    """
    from safetensors import safe_open

    f = safe_open(str(ckpt_path), framework="pt", device="cpu")
    metadata = f.metadata() or {}
    fmt = metadata.get("format")
    if fmt != "ltx-inflated-io-v1":
        raise ValueError(
            f"expected an inflate-mode checkpoint (format='ltx-inflated-io-v1'), "
            f"got format={fmt!r} at {ckpt_path}"
        )

    decoder_sd, conv_in_sd = {}, {}
    for key in f.keys():
        tensor = f.get_tensor(key)
        if key.startswith("decoder."):
            decoder_sd[key[len("decoder."):]] = tensor
        elif key.startswith("encoder_conv_in."):
            conv_in_sd[key[len("encoder_conv_in."):]] = tensor

    if not decoder_sd:
        raise ValueError(f"no decoder.* weights found in {ckpt_path}")
    _strict_load(vae.decoder, decoder_sd, "decoder")
    if conv_in_sd:
        _strict_load(vae.encoder.conv_in, conv_in_sd, "encoder.conv_in")
    vae.to(device=device, dtype=dtype)

    if verbose:
        print(
            f"[windinet] loaded inflated VAE checkpoint: {ckpt_path} "
            f"(epoch={metadata.get('epoch', '?')}, step={metadata.get('step', '?')})"
        )
    return metadata


def latent_space_fingerprint(
    ckpt_path: str | Path,
    channel_mean: list[float],
    channel_std: list[float],
    normalization_clip: float,
) -> str:
    """A short digest of everything that determines the latents a VAE produces.

    Two VAE checkpoints yielding the same fingerprint encode a given physical
    field to bit-identical latents, so precomputed latents (and any DiT trained
    on them) stay valid across the swap. A different fingerprint means the latent
    space moved and everything downstream is stale.

    Only the *encoder* side is hashed -- the grown ``encoder_conv_in`` weights
    plus the input normalization -- because those alone map fields to latents.
    The decoder is deliberately excluded: a decoder-only refinement improves
    reconstruction without invalidating a single precomputed latent, and must
    not trip the guard.

    Paths and checkpoint metadata are not used: a path can be overwritten in
    place, and two separate runs both saving "epoch010" carry identical metadata.
    """
    import hashlib

    from safetensors import safe_open

    digest = hashlib.sha256()
    with safe_open(str(ckpt_path), framework="pt", device="cpu") as f:
        conv_in_keys = sorted(k for k in f.keys() if k.startswith("encoder_conv_in."))
        if not conv_in_keys:
            # Adapter-mode checkpoints route channels through in_adapter instead.
            conv_in_keys = sorted(k for k in f.keys() if k.startswith("in_adapter."))
        if not conv_in_keys:
            raise ValueError(
                f"{ckpt_path} has neither encoder_conv_in.* nor in_adapter.* weights, so the "
                "latent space it produces cannot be fingerprinted"
            )
        for key in conv_in_keys:
            digest.update(key.encode())
            digest.update(f.get_tensor(key).float().cpu().numpy().tobytes())

    # The normalization is part of the encoder: the VAE only ever sees
    # (x - mean) / (std * clip), so changing these remaps the latents too.
    digest.update(repr([round(v, 12) for v in channel_mean]).encode())
    digest.update(repr([round(v, 12) for v in channel_std]).encode())
    digest.update(repr(round(float(normalization_clip), 12)).encode())
    return digest.hexdigest()[:16]


def _load_safetensors_vae(ckpt_path: str | Path):
    """Load a .safetensors VAE checkpoint. Returns (tensors_dict, metadata_dict)."""
    from safetensors import safe_open

    f = safe_open(str(ckpt_path), framework="pt", device="cpu")
    metadata = f.metadata() or {}

    # Separate decoder weights from adapter weights
    decoder_sd, in_sd, out_sd = {}, {}, {}
    for key in f.keys():
        tensor = f.get_tensor(key)
        if key.startswith("decoder."):
            decoder_sd[key[len("decoder."):]] = tensor
        elif key.startswith("in_adapter."):
            in_sd[key[len("in_adapter."):]] = tensor
        elif key.startswith("out_adapter."):
            out_sd[key[len("out_adapter."):]] = tensor

    # Parse metadata strings back to proper types
    import ast
    channels = ast.literal_eval(
        metadata.get(
            "channels",
            "['density','momentum_x','momentum_y','pressure']",
        )
    )
    n = int(metadata.get("n", len(channels)))
    k = int(metadata.get("k", "0"))
    activation = metadata.get("activation", "gelu")
    default_temb = float(metadata.get("default_temb", "0.0"))
    identity_init = metadata.get("identity_init", "False") == "True"

    ckpt = {
        "channels": channels, "n": n, "k": k, "activation": activation,
        "default_temb": default_temb,
        "identity_init": identity_init,
        "in_adapter": in_sd,
        "out_adapter": out_sd,
    }
    if metadata.get("normalization"):
        ckpt["normalization"] = metadata["normalization"]
        ckpt["channel_mean"] = ast.literal_eval(metadata["channel_mean"])
        ckpt["channel_std"] = ast.literal_eval(metadata["channel_std"])
        ckpt["normalization_clip"] = float(metadata["normalization_clip"])
    if decoder_sd:
        ckpt["decoder"] = decoder_sd
    return ckpt


def _load_pt_vae(ckpt_path: str | Path):
    """Load a legacy .pt VAE checkpoint."""
    return torch.load(str(ckpt_path), map_location="cpu", weights_only=False)


def load_adapted_vae(
    vae,
    ckpt_path: str | Path | None = None,
    device: str = str(get_default_device()),
    dtype: torch.dtype = torch.float32,
    *,
    channels: list[str] | None = None,
    k: int | None = None,
    activation: str | None = None,
    identity_init: bool = False,
    default_temb: float = AdaptedVAE.DEFAULT_INFERENCE_TEMB,
    verbose: bool = True,
) -> tuple[AdaptedVAE, dict[str, Any]]:
    """
    Load or create an AdaptedVAE.

    Supports both .safetensors and legacy .pt checkpoint formats.
    If ckpt_path is None, creates passthrough adapters (k=0).
    """
    ckpt = None
    if ckpt_path is not None:
        path = str(ckpt_path)
        if path.endswith(".safetensors"):
            ckpt = _load_safetensors_vae(path)
        else:
            ckpt = _load_pt_vae(path)

    channels = list((ckpt or {}).get("channels", channels or ["density", "momentum_x", "momentum_y", "pressure"]))
    n = int((ckpt or {}).get("n", len(channels)))
    k = int((ckpt or {}).get("k", 0 if k is None else k))
    activation = str((ckpt or {}).get("activation", activation or "gelu"))
    # A checkpoint's stored mode wins so its forward matches how it was trained.
    identity_init = bool((ckpt or {}).get("identity_init", identity_init))
    has_decoder = isinstance((ckpt or {}).get("decoder", None), dict)

    if default_temb == AdaptedVAE.DEFAULT_INFERENCE_TEMB and ckpt is not None:
        ckpt_temb = ckpt.get("default_temb", None)
        if ckpt_temb is not None:
            default_temb = float(ckpt_temb)

    if verbose:
        print(f"[windinet] channels={channels}, n={n}, k={k}, activation={activation}")
        if ckpt_path:
            print(f"[windinet] loading adapter checkpoint: {ckpt_path}")

    in_adapt = InAdapter(n=n, k=k, activation=activation, identity_init=identity_init).to(device=device, dtype=dtype)
    out_adapt = OutAdapter(n=n, k=k, activation=activation, identity_init=identity_init).to(device=device, dtype=dtype)

    if ckpt is not None:
        if ckpt.get("in_adapter"):
            _strict_load(in_adapt, ckpt["in_adapter"], "in_adapter")
        if ckpt.get("out_adapter"):
            _strict_load(out_adapt, ckpt["out_adapter"], "out_adapter")

    model = AdaptedVAE(
        vae=vae, in_adapter=in_adapt, out_adapter=out_adapt,
        channels=channels, k=k, default_temb=default_temb,
    ).to(device)

    if ckpt is not None and has_decoder:
        decoder_mod = vae.decoder if hasattr(vae, "decoder") else vae
        _strict_load(decoder_mod, ckpt["decoder"], "decoder")
        if verbose:
            print("[windinet] loaded finetuned decoder weights")

    meta = {
        "channels": channels, "n": n, "k": k, "activation": activation,
        "identity_init": identity_init,
        "decoder_loaded": bool(ckpt is not None and has_decoder),
        "default_temb": float(default_temb),
    }
    for key in ("normalization", "channel_mean", "channel_std", "normalization_clip"):
        if ckpt is not None and key in ckpt:
            meta[key] = ckpt[key]
    return model, meta
