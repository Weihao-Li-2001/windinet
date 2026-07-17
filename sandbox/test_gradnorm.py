"""
Test GradNorm loss weighting.

Checks:
    - initialization
    - adaptive weight update
    - weight normalization
    - stability
"""


from windinet.loss_weighting import GradNorm



def main():

    print("=" * 80)
    print("Testing GradNorm")
    print("=" * 80)



    loss_names = [
        "rmse",
        "h1",
        "ssim",
        "mlw",
    ]


    gradnorm = GradNorm(
        loss_names=loss_names,
        alpha=1.5,
        weight_lr=0.025,
    )



    # --------------------------------------------------
    # Initial weights
    # --------------------------------------------------

    print("\nInitial weights")

    weights = gradnorm.get_weights()


    for k, v in weights.items():

        print(
            f"{k:5s}: {v:.6f}"
        )



    # --------------------------------------------------
    # Simulate first epoch
    #
    # MLW has much larger value
    # and gradient contribution
    # --------------------------------------------------

    losses_epoch1 = {

        "rmse": 1.4,
        "h1": 7.8,
        "ssim": 0.99,
        "mlw": 50.0,

    }


    grad_norms_epoch1 = {

        "rmse": 0.002,
        "h1": 0.02,
        "ssim": 0.001,
        "mlw": 40.0,

    }


    gradnorm.update(
        losses=losses_epoch1,
        grad_norms=grad_norms_epoch1,
    )


    print("\nAfter first update")

    weights = gradnorm.get_weights()

    for k, v in weights.items():

        print(
            f"{k:5s}: {v:.6f}"
        )



    print(
        "weight sum:",
        sum(weights.values())
    )



    # --------------------------------------------------
    # Simulate second epoch
    #
    # Assume MLW improves slower
    # --------------------------------------------------

    losses_epoch2 = {

        "rmse": 0.8,
        "h1": 5.0,
        "ssim": 0.8,
        "mlw": 45.0,

    }


    grad_norms_epoch2 = {

        "rmse": 0.003,
        "h1": 0.03,
        "ssim": 0.002,
        "mlw": 35.0,

    }



    old_weights = gradnorm.get_weights()


    gradnorm.update(
        losses=losses_epoch2,
        grad_norms=grad_norms_epoch2,
    )


    new_weights = gradnorm.get_weights()



    print("\nAfter second update")

    for k, v in new_weights.items():

        print(
            f"{k:5s}: {v:.6f}"
        )



    print(
        "weight sum:",
        sum(new_weights.values())
    )



    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------


    assert all(
        v > 0
        for v in new_weights.values()
    ), "Negative weight detected"


    assert abs(
        sum(new_weights.values()) - 4
    ) < 1e-5, "Weights are not normalized"


    assert old_weights != new_weights, (
        "Weights did not update"
    )


    print(
        "\nAll GradNorm tests passed."
    )



if __name__ == "__main__":

    main()