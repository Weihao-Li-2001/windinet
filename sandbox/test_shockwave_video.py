import torch

from windinet.training.shockwave_data import (
    ShockWaveDataset,
    build_shockwave_video,
)

dataset = ShockWaveDataset(
    "sandbox/train.h5"
)

sample = dataset[0]

print("=== Dataset ===")
print("density:", sample["density"].shape)
print("momentum_x:", sample["momentum_x"].shape)
print("momentum_y:", sample["momentum_y"].shape)
print("pressure:", sample["pressure"].shape)

video = build_shockwave_video(
    sample,
    device=torch.device("cpu"),
)

print("\n=== Video ===")
print(video.shape)