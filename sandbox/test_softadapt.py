"""
Test SoftAdapt loss weighting.

Checks:
    - initialization
    - first update behavior
    - adaptive weight update
    - normalization
"""

from windinet.loss_weighting import SoftAdapt



def main():

    print("=" * 80)
    print("Testing SoftAdapt")
    print("=" * 80)


    loss_names = [
        "rmse",
        "h1",
        "ssim",
        "mlw",
    ]


    softadapt = SoftAdapt(
        loss_names=loss_names,
        temperature=0.1,
        normalize=True,
    )


    # --------------------------------------------------
    # Initial weights
    # --------------------------------------------------

    print("\nInitial weights")

    weights = softadapt.get_weights()

    for k, v in weights.items():

        print(
            f"{k:5s}: {v:.6f}"
        )


    # --------------------------------------------------
    # Epoch 1
    #
    # Only initialize history
    # --------------------------------------------------

    losses_epoch1 = {

        "rmse": 1.0,
        "h1": 10.0,
        "ssim": 1.0,
        "mlw": 50.0,

    }


    softadapt.update(
        losses_epoch1
    )


    print(
        "\nAfter first update"
    )


    weights = softadapt.get_weights()


    for k, v in weights.items():

        print(
            f"{k:5s}: {v:.6f}"
        )



    # --------------------------------------------------
    # Epoch 2
    #
    # Different convergence rates
    # --------------------------------------------------

    losses_epoch2 = {

        # Fast improvement
        "rmse": 0.5,

        # Slow improvement
        "h1": 9.8,

        "ssim": 0.99,

        # Getting worse
        "mlw": 55.0,

    }


    softadapt.update(
        losses_epoch2
    )


    print(
        "\nAfter second update"
    )


    weights = softadapt.get_weights()


    for k, v in weights.items():

        print(
            f"{k:5s}: {v:.6f}"
        )



    print(
        "\nWeight sum:",
        sum(weights.values())
    )


    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------

    assert abs(
        sum(weights.values()) - 4
    ) < 1e-6, (
        "Weights are not normalized"
    )


    assert weights["rmse"] < 1.0, (
        "Fast improving loss should receive smaller weight"
    )


    assert weights["mlw"] > 1.0, (
        "Increasing loss should receive larger weight"
    )


    print(
        "\nAll SoftAdapt tests passed."
    )



if __name__ == "__main__":

    main()