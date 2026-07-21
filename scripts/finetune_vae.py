#!/usr/bin/env python
"""WinDiNet -- Finetune the VAE decoder with designed losses.

Usage:
    python scripts/finetune_vae.py configs/finetune_vae.yaml
"""

import typer
import yaml
from rich.console import Console

from windinet.config import VaeTrainerConfig
from windinet.training.vae_trainer import VaeTrainer

console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Finetune the VAE decoder with designed losses.",
)


@app.command()
def main(
    config_path: str = typer.Argument(..., help="Path to YAML configuration file"),
) -> None:
    """Finetune the VAE decoder using the provided configuration."""
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    try:
        config = VaeTrainerConfig(**config_data)
    except Exception as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(1)

    trainer = VaeTrainer(config)
    saved_path = trainer.train()
    if saved_path:
        console.print(f"\n[green]Training complete.[/green] Checkpoint: {saved_path}")


if __name__ == "__main__":
    app()
