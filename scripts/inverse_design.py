#!/usr/bin/env python
"""WinDiNet -- Inverse building layout optimization.

Optimises building positions to improve pedestrian wind comfort
using WinDiNet as a differentiable surrogate model.

Usage:
    python scripts/inverse_design.py configs/inverse_opt.yaml
"""

import typer
import yaml
from rich.console import Console

console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Optimise building layouts for pedestrian wind comfort.",
)


@app.command()
def main(
    config_path: str = typer.Argument(..., help="Path to YAML configuration file"),
    run_name: str = typer.Option("inverse_opt", help="Name for the output run directory"),
) -> None:
    """Run inverse building layout optimization."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    from windinet.checkpoints import ensure_checkpoint
    from inverse.surrogate import load_ltx_surrogate
    from inverse.optimize import optimize_footprint_for_multiple_inlets

    # Resolve checkpoints
    surrogate_cfg = cfg.get("surrogate", {})
    vae_ckpt = ensure_checkpoint(surrogate_cfg.get("vae_checkpoint", "vae_decoder"))

    # Load surrogate
    console.print("Loading WinDiNet surrogate...")

    device = cfg.get("optimization", {}).get("device", "cuda")

    # load_ltx_surrogate expects a diffusion_dir containing
    # checkpoints/model_weights_*.safetensors and
    # checkpoints/scalar_embedding_*.safetensors.
    # When using HuggingFace checkpoints, stage them into a temp dir.
    import os, tempfile, shutil
    dit_ckpt = ensure_checkpoint(surrogate_cfg.get("dit_checkpoint", "dit"))
    scalar_ckpt = ensure_checkpoint(surrogate_cfg.get("scalar_checkpoint", "scalar_embedding"))

    diffusion_dir = surrogate_cfg.get("diffusion_dir")
    if diffusion_dir is None:
        # Create temp dir with expected structure
        diffusion_dir = tempfile.mkdtemp(prefix="windinet_ckpt_")
        ckpt_subdir = os.path.join(diffusion_dir, "checkpoints")
        os.makedirs(ckpt_subdir, exist_ok=True)
        os.symlink(dit_ckpt, os.path.join(ckpt_subdir, "model_weights_step_00000.safetensors"))
        os.symlink(scalar_ckpt, os.path.join(ckpt_subdir, "scalar_embedding_step_00000.safetensors"))

    surrogate = load_ltx_surrogate(
        diffusion_dir=diffusion_dir,
        vae_adapter_ckpt=vae_ckpt,
        field_size_m=surrogate_cfg.get("field_size_m", 1100.0),
        num_inference_steps=surrogate_cfg.get("num_inference_steps", 1),
        device=device,
    )

    # Run optimization
    footprint_cfg = cfg.get("footprint", {})
    pwc_cfg = cfg.get("pwc", {})
    reg_cfg = cfg.get("regularization", {})
    opt_cfg = cfg.get("optimization", {})
    domain_cfg = cfg.get("domain", {})
    log_cfg = cfg.get("logging", {})

    footprint, run_dir = optimize_footprint_for_multiple_inlets(
        solver=surrogate,
        footprint_json_path=footprint_cfg["footprint_json_path"],
        inlet_list=cfg.get("inlets", [[15.0, 0.0]]),
        run_name=run_name,
        runs_root=log_cfg.get("runs_root", "outputs/inverse"),
        H=domain_cfg.get("H", 256),
        W=domain_cfg.get("W", 256),
        tau=domain_cfg.get("tau", 2.0),
        n_steps=opt_cfg.get("n_steps", 200),
        lr=opt_cfg.get("lr", 1.0),
        device=device,
        inlet_weights=cfg.get("inlet_weights"),
        w_move=reg_cfg.get("w_move", 1e-4),
        w_cohesion=reg_cfg.get("w_cohesion", 0.001),
        cohesion_hinge=reg_cfg.get("cohesion_hinge", 0.0),
        subdivide=footprint_cfg.get("subdivide", 1),
        objective_rect=cfg.get("objective_rect"),
        optimizer_name=opt_cfg.get("optimizer", "adam"),
        transient_frames=cfg.get("transient_frames", 10),
        pwc_tau=pwc_cfg.get("tau", 1.0),
        pwc_danger_threshold=pwc_cfg.get("danger_threshold", 15.0),
        pwc_comfort_threshold=pwc_cfg.get("comfort_threshold", 5.0),
        pwc_stagnation_threshold=pwc_cfg.get("stagnation_threshold", 1.0),
        pwc_w_danger=pwc_cfg.get("w_danger", 10.0),
        pwc_w_comfort=pwc_cfg.get("w_comfort", 1.0),
        pwc_w_stagnation=pwc_cfg.get("w_stagnation", 1.0),
        plot_every=log_cfg.get("plot_every", 1),
        write_csv_every=log_cfg.get("write_csv_every", 25),
        print_every=log_cfg.get("print_every", 10),
    )

    console.print(f"\n[green]Optimization complete.[/green] Results: {run_dir}")


if __name__ == "__main__":
    app()
