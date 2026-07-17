#!/usr/bin/env python3
"""
Test construction of the ShockWave VAE.

This test only checks that the modified architecture is built correctly.
No pretrained weights are loaded.
"""

import torch

from windinet.inference.model_loader import load_shockwave_vae


def main():

    print("=" * 80)
    print("Building ShockWave VAE...")
    print("=" * 80)

    vae = load_shockwave_vae(
        "LTXV_2B_0.9.6_DEV",
        in_channels=4,
        out_channels=4,
        dtype=torch.float32,
    )

    print("✓ Successfully created ShockWave VAE.\n")

    print("=" * 80)
    print("Configuration")
    print("=" * 80)

    print(f"in_channels  : {vae.config.in_channels}")
    print(f"out_channels : {vae.config.out_channels}")

    print("\n")

    print("=" * 80)
    print("Encoder")
    print("=" * 80)

    print(f"patch_size      : {vae.encoder.patch_size}")
    print(f"encoder channels: {vae.encoder.in_channels}")

    print(f"conv_in.weight  : {tuple(vae.encoder.conv_in.weight.shape)}")

    print("\n")

    print("=" * 80)
    print("Decoder")
    print("=" * 80)

    print(f"patch_size      : {vae.decoder.patch_size}")
    print(f"decoder channels: {vae.decoder.out_channels}")

    print(f"conv_out.weight : {tuple(vae.decoder.conv_out.weight.shape)}")


if __name__ == "__main__":
    main()