"""
Test physics-informed reconstruction losses.

Checks:
    - import
    - forward computation
    - gradient flow
    - individual loss backward compatibility
"""

import torch

from windinet.losses import (
    rmse_loss,
    h1_seminorm_loss,
    SSIMLoss,
    mlw_loss,
    reconstruction_losses,
)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("Testing reconstruction losses")
    print("=" * 80)

    # --------------------------------------------------
    # Fake video fields
    # Shape:
    # B,C,T,H,W
    #
    # C=4:
    # density
    # momentum_x
    # momentum_y
    # pressure
    # --------------------------------------------------

    B = 1
    C = 4
    T = 16
    H = 64
    W = 64


    pred = torch.randn(
        B,
        C,
        T,
        H,
        W,
        device=device,
        requires_grad=True,
    )

    target = torch.randn(
        B,
        C,
        T,
        H,
        W,
        device=device,
    )


    print("\nInput:")
    print("pred:", pred.shape)
    print("target:", target.shape)


    # --------------------------------------------------
    # Individual losses
    # --------------------------------------------------

    print("\nIndividual losses")


    rmse = rmse_loss(
        pred,
        target,
    )

    print(
        f"RMSE : {rmse.item():.6f}"
    )


    h1 = h1_seminorm_loss(
        pred,
        target,
    )

    print(
        f"H1   : {h1.item():.6f}"
    )


    ssim = SSIMLoss(
        channels=C,
        window_size=11,
        sigma=1.5,
    ).to(device)


    ssim_value = ssim(
        pred,
        target,
    )

    print(
        f"SSIM : {ssim_value.item():.6f}"
    )


    mlw = mlw_loss(
        pred,
        target,
    )

    print(
        f"MLW  : {mlw.item():.6f}"
    )


    # --------------------------------------------------
    # Combined reconstruction losses
    # --------------------------------------------------

    print("\nReconstruction loss dictionary")


    losses = reconstruction_losses(
        pred,
        target,
        ssim_module=ssim,
    )


    for name, value in losses.items():

        print(
            f"{name:5s}: "
            f"{value.item():.6f}"
        )


    # --------------------------------------------------
    # Test backward
    # --------------------------------------------------

    print("\nBackward test")


    total_loss = sum(
        value
        for value in losses.values()
    )


    total_loss.backward()


    print(
        "pred.grad exists:",
        pred.grad is not None
    )


    print(
        "grad norm:",
        pred.grad.norm().item()
    )


    # --------------------------------------------------
    # Test individual gradient compatibility
    # Needed for GradNorm
    # --------------------------------------------------

    print("\nIndividual gradient test")


    pred2 = torch.randn(
        B,
        C,
        T,
        H,
        W,
        device=device,
        requires_grad=True,
    )


    losses2 = reconstruction_losses(
        pred2,
        target,
        ssim_module=ssim,
    )


    for name, loss in losses2.items():

        grad = torch.autograd.grad(
            loss,
            pred2,
            retain_graph=True,
        )[0]


        print(
            f"{name:5s} gradient norm:",
            grad.norm().item()
        )


    print("\nAll tests passed.")


if __name__ == "__main__":
    main()