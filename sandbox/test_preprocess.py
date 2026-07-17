"""
Test preprocessing pipeline for ShockWaveNet.

Checks:
1. Dataset loading
2. Video construction
"""

import torch

from windinet.training.shockwave_data import (
    ShockWaveDataset,
    build_shockwave_video,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("=== Loading dataset ===")

    dataset = ShockWaveDataset("sandbox/train.h5") 

    sample = dataset[0]

    print(sample.keys())

    print("\n=== Building video ===")

    video = build_shockwave_video(
        sample,
        device=torch.device(DEVICE),
    )

    print(video.shape)
    print(video.dtype)
    print(f"min: {video.min().item():.4f}")
    print(f"max: {video.max().item():.4f}")
    print(f"mean: {video.mean().item():.4f}")
    print(f"std: {video.std().item():.4f}")

    print("\n=== Before encode_video ===")
    print(video.permute(0, 2, 1, 3, 4).shape)


if __name__ == "__main__":
    main()