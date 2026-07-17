"""
Test VAE trainer loss weighting pipeline.

Pipeline:

prediction
    |
reconstruction_losses
    |
loss weighting strategy
    |
weighted total loss
    |
backward


This test does NOT load VAE.
"""


import torch


from windinet.losses import (
    reconstruction_losses,
    SSIMLoss,
)


from windinet.loss_weighting import (
    FixedWeighting,
    SoftAdapt,
    GradNorm,
)



def compute_total_loss(
    losses,
    weighter,
):
    """
    Same logic as vae_trainer.py
    """

    weights = weighter.get_weights()


    total_loss = sum(
        weights[name] * value
        for name, value in losses.items()
    )

    return total_loss, weights



def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print("=" * 80)
    print("Testing VAE loss weighting pipeline")
    print("=" * 80)



    # --------------------------------------------------
    # Fake VAE reconstruction output
    # --------------------------------------------------

    pred = torch.randn(
        1,
        4,
        16,
        64,
        64,
        device=device,
        requires_grad=True,
    )


    target = torch.randn(
        1,
        4,
        16,
        64,
        64,
        device=device,
    )



    ssim = SSIMLoss(
        channels=4,
    ).to(device)



    # --------------------------------------------------
    # Reconstruction losses
    # --------------------------------------------------

    losses = reconstruction_losses(
        pred=pred,
        target=target,
        ssim_module=ssim,
    )


    print("\nLoss components:")

    for k,v in losses.items():

        print(
            f"{k:5s}: {v.item():.6f}"
        )



    # ==================================================
    # Test Fixed
    # ==================================================

    print("\n" + "=" * 40)
    print("Fixed weighting")
    print("=" * 40)


    fixed = FixedWeighting(
        {
            "rmse":1.0,
            "h1":0.5,
            "ssim":0.2,
            "mlw":0.05,
        }
    )


    total_loss, weights = compute_total_loss(
        losses,
        fixed,
    )


    print(
        "weights:",
        weights
    )


    print(
        "total loss:",
        total_loss.item()
    )


    pred.grad = None

    total_loss.backward(
        retain_graph=True
    )


    assert pred.grad is not None



    print(
        "gradient:",
        pred.grad.norm().item()
    )



    # ==================================================
    # Test SoftAdapt
    # ==================================================

    print("\n" + "=" * 40)
    print("SoftAdapt weighting")
    print("=" * 40)


    softadapt = SoftAdapt(
        loss_names=[
            "rmse",
            "h1",
            "ssim",
            "mlw",
        ],
        temperature=0.5,
    )


    # first epoch initialize

    softadapt.update(
        {
            k:v.item()
            for k,v in losses.items()
        }
    )


    # simulate next epoch

    softadapt.update(
        {
            "rmse": losses["rmse"].item()*0.9,
            "h1": losses["h1"].item()*0.99,
            "ssim": losses["ssim"].item()*0.98,
            "mlw": losses["mlw"].item()*1.05,
        }
    )


    total_loss, weights = compute_total_loss(
        losses,
        softadapt,
    )


    print(
        "weights:",
        weights
    )


    print(
        "weight sum:",
        sum(weights.values())
    )


    assert abs(
        sum(weights.values()) - 4
    ) < 1e-5



    # ==================================================
    # Test GradNorm interface
    # ==================================================

    print("\n" + "=" * 40)
    print("GradNorm interface")
    print("=" * 40)


    gradnorm = GradNorm(
        loss_names=[
            "rmse",
            "h1",
            "ssim",
            "mlw",
        ]
    )


    gradnorm.update(
        losses={
            k:v.item()
            for k,v in losses.items()
        },
        grad_norms={
            "rmse":0.01,
            "h1":0.1,
            "ssim":0.02,
            "mlw":10.0,
        }
    )


    weights = gradnorm.get_weights()


    print(
        "weights:",
        weights
    )


    print(
        "weight sum:",
        sum(weights.values())
    )


    assert abs(
        sum(weights.values()) - 4
    ) < 1e-5



    print(
        "\nAll VAE loss weighting tests passed."
    )



if __name__ == "__main__":

    main()