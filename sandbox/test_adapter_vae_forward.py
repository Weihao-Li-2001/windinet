import torch

from windinet.inference.model_loader import load_vae
from windinet.vae_adapter import load_adapted_vae


device = "cpu"


# 1. load original LTX VAE
vae = load_vae(
    "LTXV_2B_0.9.6_DEV",
    dtype=torch.float32,
)


# 2. wrap adapter
vae, meta = load_adapted_vae(
    vae,
    ckpt_path=None,
    device=device,
    dtype=torch.float32,
)


print(meta)


# 3. fake CFD video
x = torch.randn(
    1,
    3,
    17,
    128,
    128,
    device=device,
)


# 4. encode

enc = vae.encode(x)

latents = enc.latent_dist.mean

print("latent:", latents.shape)


# 5. decode

out = vae.decode(latents)

recon = out.sample


print("recon:", recon.shape)


# 6. loss

loss = torch.nn.functional.mse_loss(
    recon,
    x
)

print("loss:", loss.item())


# 7. backward

loss.backward()

print("backward OK")