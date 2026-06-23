#!/usr/bin/env python
"""Run pystaggerflow ground truth on all step layouts from an optimisation run.

Usage:
    python -m inverse.ground_truth windNet/runs/ltx_publish/final

Runs the CFD solver on each map_step_NNN.json, computes metrics matching
the surrogate objective function, and saves results to ground_truth/ subdir.

Building masks are rendered using the same sigmoid-based rasterisation as
DifferentiableFootprint (tau, discrete threshold), so the GT solver sees
**exactly** the same building footprints as the surrogate.

Output:
    ground_truth/metrics.csv           Per-step metrics (same columns as history.csv)
    ground_truth/speed_step_NNN.npy    Time-averaged speed (H, W) for every step
    ground_truth/fields_step_NNN.npz   Full (T, H, W) u,v fields for initial/final
"""

import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
#  Building mask rendering (numpy replica of DifferentiableFootprint)
# ---------------------------------------------------------------------------

def render_building_mask(map_data, H=256, W=256, tau=2.0, subdivide=1):
    """Render a binary building mask identical to DifferentiableFootprint.

    Uses sigmoid-based soft rendering then binarises at 0.5, matching
    ``DifferentiableFootprint.forward(discrete=True)``.

    Returns (H, W) uint8 array with 1=building, 0=fluid.
    """
    bounds = map_data["bounds"]["extents"]
    bx0, bx1, by0, by1 = bounds[0], bounds[1], bounds[2], bounds[3]

    xs = np.linspace(bx0, bx1, W, dtype=np.float64)
    ys = np.linspace(by0, by1, H, dtype=np.float64)
    Y, X = np.meshgrid(ys, xs, indexing="ij")  # (H, W) each

    # Collect all blocks (with optional subdivision for trainable ones)
    rects = []
    for b in map_data["blocks"]:
        ext = b["extents"]
        xmin, xmax, ymin, ymax = ext[0], ext[1], ext[2], ext[3]
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        w = xmax - xmin
        h = ymax - ymin
        is_train = bool(b.get("trainable", False))

        if is_train and subdivide > 1:
            sub_w = w / subdivide
            sub_h = h / subdivide
            for row in range(subdivide):
                for col in range(subdivide):
                    sub_cx = xmin + sub_w * (col + 0.5)
                    sub_cy = ymin + sub_h * (row + 0.5)
                    rects.append((sub_cx, sub_cy, sub_w, sub_h))
        else:
            rects.append((cx, cy, w, h))

    # Clamp centres inside bounds (matching DifferentiableFootprint)
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))

    # Accumulate soft union: 1 - prod(1 - p_i)
    log_complement = np.zeros((H, W), dtype=np.float64)  # sum of log(1 - p_i)

    for cx, cy, w, h in rects:
        cx_c = np.clip(cx, bx0 + 0.5 * w, bx1 - 0.5 * w)
        cy_c = np.clip(cy, by0 + 0.5 * h, by1 - 0.5 * h)
        xmin_r = cx_c - 0.5 * w
        xmax_r = cx_c + 0.5 * w
        ymin_r = cy_c - 0.5 * h
        ymax_r = cy_c + 0.5 * h

        px = sig((X - xmin_r) / tau) * sig((xmax_r - X) / tau)
        py = sig((Y - ymin_r) / tau) * sig((ymax_r - Y) / tau)
        p = np.clip(px * py, 0.0, 1.0)

        log_complement += np.log1p(-p + 1e-30)  # log(1 - p_i)

    building_prob = 1.0 - np.exp(log_complement)
    mask = (building_prob > 0.5).astype(np.uint8)
    return mask


# ---------------------------------------------------------------------------
#  Solver runner (executed in worker process)
# ---------------------------------------------------------------------------

def _run_solver(occupancy_map, inlet_u, inlet_v, field_size, grid_size=256,
                warmup=50, sim_steps=1796, record_every=4, temporal_stride=4):
    """Run pystaggerflow with a pre-computed occupancy map.

    The occupancy_map should be (numX, numY) = (grid_size, grid_size) with
    1 = obstacle, 0 = fluid.
    """
    from pystaggerflow.solver_utils import EulerFlowSolver

    resolution = field_size / grid_size

    # Boundary conditions (same logic as TwoDEulerFlow)
    heading = np.arctan2(inlet_v, inlet_u)
    tol = 1e-4
    if abs(heading) < tol:
        BC_spec = [0, 3, 2, 2]
    elif abs(heading - np.pi / 2) < tol:
        BC_spec = [2, 2, 0, 3]
    elif abs(abs(heading) - np.pi) < tol:
        BC_spec = [3, 0, 2, 2]
    elif abs(heading + np.pi / 2) < tol:
        BC_spec = [2, 2, 3, 0]
    elif heading > 0 and heading < np.pi / 2:
        BC_spec = [0, 3, 0, 3]
    elif heading > np.pi / 2 and heading < np.pi:
        BC_spec = [3, 0, 0, 3]
    elif heading < 0 and heading > -np.pi / 2:
        BC_spec = [0, 3, 3, 0]
    else:
        BC_spec = [3, 0, 3, 0]

    # EulerFlowSolver.init_grid expects a 3D occupancy map (X, Y, Z) and
    # slices [:,:,0].  Add a dummy Z axis to our 2D mask.
    if occupancy_map.ndim == 2:
        occupancy_map = occupancy_map[:, :, np.newaxis]

    solver = EulerFlowSolver(overRelaxation=1.5)
    solver.init_solution(
        occupancy_map, inlet_u, inlet_v,
        density=0.60, h=resolution, dt=0.25, BC_spec=BC_spec,
    )

    # Initial solve (same as TwoDEulerFlow.__init__ calling update_wind)
    solver.solve(200)

    # Warmup
    for _ in range(warmup):
        solver.solve(200)

    # Simulate and record
    u_frames, v_frames = [], []
    numXg = occupancy_map.shape[0] + 2
    numYg = occupancy_map.shape[1] + 2
    for step in range(sim_steps):
        solver.solve(200)
        if step % record_every == 0:
            us = solver.u.reshape(numXg, numYg).T  # (numYg, numXg)
            vs = solver.v.reshape(numXg, numYg).T
            u_frames.append(us[1:-1, 1:-1].copy())
            v_frames.append(vs[1:-1, 1:-1].copy())

    u_seq = np.stack(u_frames)[::temporal_stride].astype(np.float32)
    v_seq = np.stack(v_frames)[::temporal_stride].astype(np.float32)

    return u_seq, v_seq  # (T, H, W) each


# ---------------------------------------------------------------------------
#  Metrics (numpy equivalents of objective.py)
# ---------------------------------------------------------------------------

def _masked_mean(x, mask):
    """x: (..., H, W), mask: (H, W) -> scalar."""
    while mask.ndim < x.ndim:
        mask = mask[np.newaxis]
    return (x * mask).sum(axis=(-2, -1)) / (mask.sum(axis=(-2, -1)) + 1e-6)


def _masked_variance(x, mask):
    m = _masked_mean(x, mask)
    while m.ndim < x.ndim:
        m = m[..., np.newaxis]
    return _masked_mean((x - m) ** 2, mask)


def _compute_vorticity(u, v, dx=1.0, dy=1.0):
    """Central-difference vorticity matching objective.py."""
    dv_dx = (v[..., :, 2:] - v[..., :, :-2]) / (2 * dx)
    du_dy = (u[..., 2:, :] - u[..., :-2, :]) / (2 * dy)
    dv_dx = dv_dx[..., 1:-1, :]
    du_dy = du_dy[..., :, 1:-1]
    return dv_dx - du_dy


def compute_metrics(u, v, obj_mask, config):
    """Compute metrics matching urban_wind_loss from objective.py.

    Returns dict with same keys as history.csv.
    """
    speed = np.sqrt(u ** 2 + v ** 2 + 1e-8)

    mean_speed = float(_masked_mean(speed, obj_mask).mean())
    std_val = float(_masked_variance(speed, obj_mask).mean())

    vort = _compute_vorticity(u, v)
    vort_mask = obj_mask[1:-1, 1:-1]
    vort_loss = float(_masked_mean(vort ** 2, vort_mask).mean())

    # CVaR
    flat_mask = np.broadcast_to(obj_mask[None], speed.shape)
    active = speed[flat_mask > 0.5]
    sorted_vals = np.sort(active)
    cutoff = int(len(sorted_vals) * config.get("cvar_quantile", 0.9))
    cvar = float(sorted_vals[cutoff:].mean()) if cutoff < len(sorted_vals) else 0.0

    # Exceedance (hard threshold for ground truth -- no sigmoid smoothing)
    thr = config.get("exceedance_threshold", 8.0)
    exceedance = float((active > thr).mean())

    # Loss values (matching config weights)
    target = config.get("target_mean_speed", 5.0)
    w_mean = config.get("w_mean", 1.0)
    w_std = config.get("w_std", 1.0)
    w_vort = config.get("w_vorticity", 0.3)
    w_cvar = config.get("w_cvar", 0.0)
    w_exc = config.get("w_exceedance", 0.0)

    loss_flow = (
        w_mean * (mean_speed - target) ** 2
        + w_std * std_val
        + w_vort * vort_loss
    )
    loss_total = loss_flow + w_cvar * cvar + w_exc * exceedance

    return {
        "mean_speed": mean_speed,
        "std": std_val,
        "vorticity": vort_loss,
        "cvar": cvar,
        "exceedance": exceedance,
        "loss_flow": loss_flow,
        "loss_total": loss_total,
    }


# ---------------------------------------------------------------------------
#  Objective mask
# ---------------------------------------------------------------------------

def _compute_downstream_mask(map_data, config, H=256, W=256):
    """Binary rectangular mask for the downstream objective region."""
    bounds = map_data["bounds"]["extents"][:4]
    bx0, bx1, by0, by1 = bounds
    sx = (W - 1) / (bx1 - bx0 + 1e-12)
    sy = (H - 1) / (by1 - by0 + 1e-12)

    dw = config.get("downstream_width", 50)
    dp = config.get("downstream_pad", 20)
    gap = 5.0

    right_edges, top_edges, bot_edges = [], [], []
    for b in map_data["blocks"]:
        if b.get("trainable", False):
            xmin, xmax, ymin, ymax = b["extents"][:4]
            right_edges.append(xmax)
            top_edges.append(ymax)
            bot_edges.append(ymin)

    if not right_edges:
        raise ValueError("No trainable buildings found in map")

    x0 = (max(right_edges) - bx0) * sx + gap
    x1 = min(x0 + dw, W - 1)
    y0 = max((min(bot_edges) - by0) * sy - dp, 0)
    y1 = min((max(top_edges) - by0) * sy + dp, H - 1)

    mask = np.zeros((H, W), dtype=np.float32)
    mask[int(y0):int(y1), int(x0):int(x1)] = 1.0
    return mask


def _get_objective_mask(map_data, config, H=256, W=256):
    """Get the objective region mask."""
    obj_rect = config.get("objective_rect")
    if obj_rect is not None:
        x0, x1, y0, y1 = [int(v) for v in obj_rect]
        mask = np.zeros((H, W), dtype=np.float32)
        mask[y0:y1, x0:x1] = 1.0
        return mask
    if config.get("objective_region") == "downstream":
        return _compute_downstream_mask(map_data, config, H, W)
    raise ValueError(f"Unknown objective_region: {config.get('objective_region')}")


# ---------------------------------------------------------------------------
#  Worker function
# ---------------------------------------------------------------------------

def _process_step(args):
    """Worker: run solver + compute metrics for one step layout."""
    (step_idx, occupancy_map, inlet_u, inlet_v, field_size,
     H, W, obj_mask, config, save_fields) = args

    t0 = time.time()
    u, v = _run_solver(occupancy_map, inlet_u, inlet_v, field_size, grid_size=H)
    elapsed = time.time() - t0

    metrics = compute_metrics(u, v, obj_mask, config)
    metrics["step"] = step_idx

    # time-averaged speed
    speed = np.sqrt(u ** 2 + v ** 2)
    avg_speed = speed.mean(axis=0)  # (H, W)

    result = {
        "step": step_idx,
        "metrics": metrics,
        "avg_speed": avg_speed,
        "elapsed": elapsed,
    }
    if save_fields:
        result["u"] = u
        result["v"] = v

    return result


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def run_ground_truth(run_dir, max_workers=8):
    """Run pystaggerflow ground truth on all step layouts."""
    run_dir = Path(run_dir)
    config = json.load(open(run_dir / "config.json"))
    H = config.get("H", 256)
    W = config.get("W", 256)
    tau = config.get("tau", 2.0)
    subdivide = config.get("subdivide", 1)

    inlet_list = config["inlet_list"]
    inlet_u, inlet_v = inlet_list[0]  # use first inlet

    # field size from map bounds
    init_map = json.load(open(run_dir / "steps" / "map_step_000.json"))
    bounds = init_map["bounds"]["extents"]
    field_size = bounds[1] - bounds[0]

    # objective mask (from initial building positions, stays fixed)
    obj_mask = _get_objective_mask(init_map, config, H, W)

    # find all step files
    steps_dir = run_dir / "steps"
    step_files = sorted(steps_dir.glob("map_step_*.json"))
    n_steps = len(step_files)
    print(f"[gt] {n_steps} step layouts, field_size={field_size}m, "
          f"inlet=({inlet_u:.1f}, {inlet_v:.1f}), tau={tau}, "
          f"subdivide={subdivide}")
    print(f"[gt] objective mask: {int(obj_mask.sum())} pixels active")

    # output dir
    gt_dir = run_dir / "ground_truth"
    gt_dir.mkdir(exist_ok=True)

    # save mask
    np.save(gt_dir / "objective_mask.npy", obj_mask)

    # Pre-render building masks for all steps using the same sigmoid
    # rendering as the surrogate's DifferentiableFootprint
    print("[gt] rendering building masks (matching DifferentiableFootprint)...")
    step_masks = {}
    for sf in step_files:
        idx = int(sf.stem.split("_")[-1])
        map_data = json.load(open(sf))
        bldg_mask = render_building_mask(map_data, H, W, tau, subdivide)
        # Transpose (H,W) -> (W,H) = (numX, numY) for solver
        step_masks[idx] = bldg_mask.T
    print(f"[gt] rendered {len(step_masks)} building masks")

    # Quick sanity: report pixel count for first/last
    for idx in [0, max(step_masks.keys())]:
        if idx in step_masks:
            npx = step_masks[idx].sum()
            print(f"  step {idx}: {npx} building pixels")

    # prepare worker args
    # save full fields for first and last step (for speed comparison figure)
    save_steps = {0, n_steps - 1}
    args_list = []
    for sf in step_files:
        idx = int(sf.stem.split("_")[-1])
        args_list.append((
            idx, step_masks[idx], inlet_u, inlet_v, field_size,
            H, W, obj_mask, config, idx in save_steps,
        ))

    # run in parallel
    t_start = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_step, a): a[0] for a in args_list}
        for future in as_completed(futures):
            step_idx = futures[future]
            try:
                res = future.result()
                results[res["step"]] = res
                m = res["metrics"]
                print(f"  step {res['step']:3d}: "
                      f"flow_loss={m['loss_flow']:.2f}  "
                      f"mean_spd={m['mean_speed']:.2f}  "
                      f"cvar={m['cvar']:.2f}  "
                      f"exc={m['exceedance']:.3f}  "
                      f"({res['elapsed']:.0f}s)")
            except Exception as e:
                print(f"  step {step_idx}: FAILED -- {e}")

    total_time = time.time() - t_start
    print(f"\n[gt] completed {len(results)}/{n_steps} in {total_time:.0f}s")

    # sort by step index
    sorted_steps = sorted(results.keys())

    # ---- save CSV ----
    csv_path = gt_dir / "metrics.csv"
    fieldnames = ["step", "loss_flow", "loss_total", "mean_speed",
                   "std", "vorticity", "cvar", "exceedance"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in sorted_steps:
            writer.writerow(results[s]["metrics"])
    print(f"[gt] saved {csv_path}")

    # ---- save speed fields ----
    for s in sorted_steps:
        res = results[s]
        np.save(gt_dir / f"speed_step_{s:03d}.npy", res["avg_speed"])
        if res.get("u") is not None:
            np.savez_compressed(
                gt_dir / f"fields_step_{s:03d}.npz",
                u=res["u"], v=res["v"],
            )
            print(f"[gt] saved full fields for step {s}")

    # ---- save building masks for reference ----
    for s in sorted_steps:
        # Save in (H, W) orientation for easy comparison with surrogate
        np.save(gt_dir / f"bldg_mask_step_{s:03d}.npy", step_masks[s].T)

    print(f"[gt] all results saved to {gt_dir}")
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m inverse.ground_truth <run_dir>")
        sys.exit(1)
    run_ground_truth(sys.argv[1], max_workers=8)
