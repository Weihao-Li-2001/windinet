"""
Test gradient norm calculation for GradNorm.

Checks:

    individual loss
            |
            v
    compute_grad_norms()
            |
            v
    gradient statistics


    then:
            |
            v
    total loss backward
"""


import torch
import torch.nn as nn


from windinet.loss_weighting.utils import (
    compute_grad_norms,
)



class DummyDecoder(nn.Module):
    """
    Small network mimicking VAE decoder parameters.
    """

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )


    def forward(self, x):

        return self.net(x)



def main():


    print("=" * 80)
    print("Testing gradient norm calculation")
    print("=" * 80)



    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    # --------------------------------------------------
    # Dummy decoder
    # --------------------------------------------------

    model = DummyDecoder().to(device)



    # --------------------------------------------------
    # Fake input/output
    # --------------------------------------------------

    x = torch.randn(
        8,
        16,
        device=device,
    )


    target = torch.randn(
        8,
        16,
        device=device,
    )


    pred = model(x)



    # --------------------------------------------------
    # Individual losses
    # --------------------------------------------------

    losses = {

        "rmse":
            torch.sqrt(
                torch.mean(
                    (pred - target) ** 2
                )
            ),


        "h1":
            torch.mean(
                torch.abs(
                    pred[:, 1:]
                    -
                    pred[:, :-1]
                )
            ),


        "ssim":
            1.0 -
            torch.mean(
                pred * target
            ),


        "mlw":
            torch.mean(
                (pred - target) ** 2
            ) * 10.0,

    }



    print("\nLosses:")

    for k,v in losses.items():

        print(
            f"{k:5s}: {v.item():.6f}"
        )



    # --------------------------------------------------
    # Compute gradient norms
    # --------------------------------------------------

    grad_norms = compute_grad_norms(
        losses=losses,
        parameters=model.parameters(),
    )



    print("\nGradient norms:")

    for k,v in grad_norms.items():

        print(
            f"{k:5s}: {v:.8f}"
        )



    # --------------------------------------------------
    # Checks
    # --------------------------------------------------

    assert len(grad_norms) == 4


    assert all(
        value > 0
        for value in grad_norms.values()
    ), (
        "Gradient norm should be positive"
    )



    # --------------------------------------------------
    # Check backward still works
    # --------------------------------------------------

    total_loss = sum(
        losses.values()
    )


    total_loss.backward()



    grad_norm = 0.0


    for p in model.parameters():

        if p.grad is not None:

            grad_norm += (
                p.grad.norm().item()
            )


    print(
        "\nBackward gradient norm:",
        grad_norm
    )



    assert grad_norm > 0



    print(
        "\nAll gradient norm tests passed."
    )



if __name__ == "__main__":

    main()