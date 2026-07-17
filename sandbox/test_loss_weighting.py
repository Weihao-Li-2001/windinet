"""
Test loss weighting strategies.

Checks:
    - FixedWeighting initialization
    - weight retrieval
    - weighted loss computation
    - gradient flow
"""


import torch


from windinet.loss_weighting import (
    FixedWeighting,
)


from windinet.losses import (
    reconstruction_losses,
    SSIMLoss,
)



def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )


    print("=" * 80)
    print("Testing loss weighting framework")
    print("=" * 80)



    # --------------------------------------------------
    # Fake prediction and target
    # --------------------------------------------------

    B = 1
    C = 4
    T = 8
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



    # --------------------------------------------------
    # Compute individual losses
    # --------------------------------------------------

    ssim = SSIMLoss(
        channels=C,
    ).to(device)



    losses = reconstruction_losses(
        pred,
        target,
        ssim_module=ssim,
    )


    print("\nLoss components:")

    for name, value in losses.items():

        print(
            f"{name:5s}: {value.item():.6f}"
        )



    # --------------------------------------------------
    # Build fixed weighting strategy
    # --------------------------------------------------

    weighting = FixedWeighting(
        weights={
            "rmse": 1.0,
            "h1": 0.5,
            "ssim": 0.2,
            "mlw": 0.05,
        }
    )


    weights = weighting.get_weights()



    print("\nWeights:")

    for name, weight in weights.items():

        print(
            f"{name:5s}: {weight}"
        )



    # --------------------------------------------------
    # Apply weighting
    # --------------------------------------------------

    total_loss = torch.tensor(
        0.0,
        device=device,
    )


    for name, loss_value in losses.items():

        total_loss += (
            weights[name]
            *
            loss_value
        )


    print(
        "\nWeighted total loss:",
        total_loss.item()
    )



    # --------------------------------------------------
    # Backward
    # --------------------------------------------------

    total_loss.backward()


    print(
        "\nBackward test:"
    )


    print(
        "pred.grad exists:",
        pred.grad is not None,
    )


    print(
        "grad norm:",
        pred.grad.norm().item()
    )



    # --------------------------------------------------
    # Test update interface
    # --------------------------------------------------

    weighting.update(
        losses={
            k: v.item()
            for k, v in losses.items()
        }
    )


    print(
        "\nUpdate interface passed."
    )


    print(
        "\nAll tests passed."
    )



if __name__ == "__main__":

    main()