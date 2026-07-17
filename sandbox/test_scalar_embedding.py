"""
Test scalar embedding.
"""

import torch

from windinet.training.shockwave_data import (
    ShockWaveDataset,
    extract_scalars,
)
from windinet.scalar_embeddings import ScalarEmbedder


def main():
    dataset = ShockWaveDataset("sandbox/train.h5")
    sample = dataset[0]

    print("=== Metadata ===")
    print(sample["meta"])

    scalars = extract_scalars(
        sample["meta"],
        ["gamma"],
    )

    print("\n=== Extracted Scalars ===")
    print(scalars)

    scalar_tensor = torch.tensor(
        [[scalars["gamma"]]],
        dtype=torch.float32,
    )

    print("\nScalar tensor:")
    print(scalar_tensor)
    print(scalar_tensor.shape)

    embedder = ScalarEmbedder(
        input_dim=1,
        embed_dim=768,     # 根据你的DiT hidden size修改
    )

    embedding = embedder(scalar_tensor)

    print("\n=== Embedding ===")
    print(embedding.shape)


if __name__ == "__main__":
    main()