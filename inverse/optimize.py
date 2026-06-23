"""Gradient-based building layout optimiser with pedestrian wind comfort objective."""
import csv
import math
import os
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inverse.footprint import DifferentiableFootprint
from inverse.logger import OptimizationLogger
from inverse.objective import pwc_loss


# ======================================================================
# Rotation helpers -- the LTX surrogate was trained with the inlet always
# from the left.  For other wind directions we rotate the building mask
# so that the wind effectively enters from the left, run the surrogate,
# and then rotate the output velocity fields back.
# ======================================================================

def _inlet_rotation(u, v):
    """Compute rotation index and speed for an inlet direction.

    Returns (speed, k_rot) where *k_rot* is the number of 90 deg CCW image
    rotations to apply so that the wind enters from the left.

    Only multiples of 90 deg are supported.
    """
    u, v = float(u), float(v)
    speed = math.sqrt(u ** 2 + v ** 2)
    angle_deg = math.degrees(math.atan2(v, u))
    k_rot = round(-angle_deg / 90) % 4
    # Warn if the angle isn't close to a multiple of 90 deg
    nearest = -k_rot * 90
    if abs(((angle_deg - nearest + 180) % 360) - 180) > 5:
        import warnings
        warnings.warn(
            f"Inlet ({u}, {v}) angle={angle_deg:.1f} deg is not near a "
            f"multiple of 90 deg; snapping to k_rot={k_rot}."
        )
    return speed, k_rot


def _rotate_velocity_back(u_rot, v_rot, k_rot):
    """Transform velocity fields from rotated frame back to original.

    Applies spatial rotation (putting pixels back to their original
    positions) and velocity component transformation (adjusting the
    physical velocity directions).

    Parameters
    ----------
    u_rot, v_rot : Tensor (..., H, W) -- physical (u, v) in the rotated frame.
    k_rot : int (0-3), number of CCW 90 deg rotations that were applied.

    Returns
    -------
    u_orig, v_orig : Tensor (..., H, W)
    """
    u_back = torch.rot90(u_rot, k=-k_rot, dims=(-2, -1))
    v_back = torch.rot90(v_rot, k=-k_rot, dims=(-2, -1))

    if k_rot == 0:
        return u_back, v_back
    elif k_rot == 1:
        return v_back, -u_back
    elif k_rot == 2:
        return -u_back, -v_back
    elif k_rot == 3:
        return -v_back, u_back
    else:
        raise ValueError(f"Invalid k_rot: {k_rot}")


def _make_unique_run_dir(root_dir: str, run_name: str) -> str:
    os.makedirs(root_dir, exist_ok=True)
    base = os.path.join(root_dir, run_name)

    if not os.path.exists(base):
        os.makedirs(base, exist_ok=False)
        return base

    i = 1
    while True:
        cand = f"{base}_{i:03d}"
        if not os.path.exists(cand):
            os.makedirs(cand, exist_ok=False)
            return cand
        i += 1


def _footprint_to_map_dict(footprint, original_map_data: dict) -> dict:
    """Export current footprint state to a pystaggerflow-compatible dict.

    For subdivide>1, the parent block extents are set to the bounding box of
    all sub-blocks, and each sub-block is also recorded in a ``"sub_blocks"``
    list on the parent so that visualisation code can render them individually.
    """
    data = json.loads(json.dumps(original_map_data))  # deep copy
    centers = footprint.centers.detach().cpu().numpy()
    sizes = footprint.sizes.detach().cpu().numpy()
    parent_idx = footprint.parent_idx.cpu().numpy()

    if footprint.subdivide <= 1:
        # 1:1 mapping
        for i, b in enumerate(data["blocks"]):
            cx, cy = float(centers[i, 0]), float(centers[i, 1])
            w, h = float(sizes[i, 0]), float(sizes[i, 1])
            b["extents"][:4] = [cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2]
            b.pop("sub_blocks", None)
    else:
        n_blocks = len(data["blocks"])
        for pid in range(n_blocks):
            sel = parent_idx == pid
            b = data["blocks"][pid]
            if not sel.any():
                b.pop("sub_blocks", None)
                continue
            sub_c = centers[sel]   # (K, 2)
            sub_s = sizes[sel]     # (K, 2)
            # store individual sub-blocks for visualisation
            sub_blocks = []
            for j in range(len(sub_c)):
                cx, cy = float(sub_c[j, 0]), float(sub_c[j, 1])
                sw, sh = float(sub_s[j, 0]), float(sub_s[j, 1])
                sub_blocks.append([cx - sw/2, cx + sw/2, cy - sh/2, cy + sh/2])
            b["sub_blocks"] = sub_blocks
            # parent extents = bounding box of all sub-blocks
            all_x0 = min(s[0] for s in sub_blocks)
            all_x1 = max(s[1] for s in sub_blocks)
            all_y0 = min(s[2] for s in sub_blocks)
            all_y1 = max(s[3] for s in sub_blocks)
            b["extents"][:4] = [all_x0, all_x1, all_y0, all_y1]

    return data


def optimize_footprint_for_multiple_inlets(
    solver,
    footprint_json_path,
    inlet_list,
    run_name: str,
    runs_root: str = "runs",
    original_map_path=None,
    H=256,
    W=256,
    tau=2.0,
    n_steps=50,
    lr=0.05,
    device="cuda" if torch.cuda.is_available() else "cpu",
    inlet_weights=None,
    w_move=1e-4,
    w_cohesion=0.0,
    cohesion_hinge=0.0,
    subdivide=1,
    objective_rect=None,
    optimizer_name="adam",
    optimizer_kwargs=None,
    transient_frames=0,
    pwc_tau=1.0,
    pwc_danger_threshold=15.0,
    pwc_comfort_threshold=5.0,       # scalar or list[float] (one per inlet)
    pwc_stagnation_threshold=0.5,    # scalar or list[float] (one per inlet)
    pwc_w_danger=10.0,
    pwc_w_comfort=1.0,
    pwc_w_stagnation=0.3,
    plot_every=1,
    write_csv_every=25,
    print_every=10,
):
    # ---- create unique run directory ----
    run_dir = _make_unique_run_dir(runs_root, run_name)
    steps_dir = os.path.join(run_dir, "steps")
    os.makedirs(steps_dir, exist_ok=True)

    # load original map data for per-step JSON export
    original_map_data = json.loads(open(footprint_json_path).read())

    # ---- CSV file for per-step metrics ----
    csv_path = os.path.join(run_dir, "history.csv")
    csv_fields = [
        "step", "loss_total", "loss_flow", "move_pen",
        "mean_speed", "danger", "comfort", "stagnation",
    ]
    csv_f = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_f, fieldnames=csv_fields)
    csv_writer.writeheader()

    # save config for reproducibility
    config = dict(
        footprint_json_path=footprint_json_path,
        original_map_path=str(original_map_path) if original_map_path else footprint_json_path,
        inlet_list=inlet_list,
        H=int(H), W=int(W),
        tau=float(tau),
        n_steps=int(n_steps),
        lr=float(lr),
        device=str(device),
        inlet_weights=None if inlet_weights is None else list(inlet_weights),
        w_move=float(w_move),
        w_cohesion=float(w_cohesion),
        cohesion_hinge=float(cohesion_hinge),
        subdivide=int(subdivide),
        objective_rect=list(objective_rect) if objective_rect is not None else None,
        optimizer_name=str(optimizer_name),
        optimizer_kwargs=optimizer_kwargs,
        transient_frames=int(transient_frames),
        pwc_tau=float(pwc_tau),
        pwc_danger_threshold=float(pwc_danger_threshold),
        pwc_comfort_threshold=(list(pwc_comfort_threshold)
                               if isinstance(pwc_comfort_threshold, (list, tuple))
                               else float(pwc_comfort_threshold)),
        pwc_stagnation_threshold=(list(pwc_stagnation_threshold)
                                  if isinstance(pwc_stagnation_threshold, (list, tuple))
                                  else float(pwc_stagnation_threshold)),
        pwc_w_danger=float(pwc_w_danger),
        pwc_w_comfort=float(pwc_w_comfort),
        pwc_w_stagnation=float(pwc_w_stagnation),
        plot_every=int(plot_every),
        write_csv_every=int(write_csv_every),
        created_unix=time.time(),
    )
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ---- visualization logger ----
    vis_logger = OptimizationLogger(run_dir, plot_every=plot_every, write_csv_every=write_csv_every)
    vis_logger.write_config(config)

    # ---- freeze solver (but keep gradients through its ops) ----
    solver = solver.to(device)
    solver.eval()
    for p in solver.parameters():
        p.requires_grad_(False)

    # ---- footprint module ----
    footprint = DifferentiableFootprint(
        footprint_json_path, H=H, W=W, tau=tau, device=device,
        subdivide=subdivide,
    ).to(device)

    # ---- inlets (with rotation for non-left wind directions) ----
    inlets = torch.tensor(inlet_list, dtype=torch.float32, device=device)  # (K,2)
    inlet_u = inlets[:, 0]  # (K,)
    inlet_v = inlets[:, 1]  # (K,)
    K = inlets.shape[0]

    # Pre-compute rotation parameters for each inlet
    rotation_params = [_inlet_rotation(float(inlet_u[k]), float(inlet_v[k]))
                       for k in range(K)]
    for k, (spd, krot) in enumerate(rotation_params):
        if krot != 0:
            print(f"[opt] inlet {k} ({float(inlet_u[k]):.1f}, {float(inlet_v[k]):.1f}) -> "
                  f"rotate {krot*90} deg CCW, speed={spd:.1f} m/s")

    if inlet_weights is None:
        inlet_weights = torch.ones(K, device=device) / K
    else:
        inlet_weights = torch.tensor(inlet_weights, dtype=torch.float32, device=device)
        inlet_weights = inlet_weights / (inlet_weights.sum() + 1e-8)

    # ---- objective region mask (fixed, does not move with buildings) ----
    if objective_rect is not None:
        x0, x1, y0, y1 = objective_rect
        obj_region = torch.zeros(H, W, device=device)
        obj_region[int(y0):int(y1), int(x0):int(x1)] = 1.0
        print(f"[opt] objective rect: x=[{x0},{x1}] y=[{y0},{y1}]  "
              f"active area={float(obj_region.sum()):.0f}/{H*W} pixels")
    else:
        obj_region = None

    # ---- save initial building JSON (step -1 = before optimisation) ----
    init_map = _footprint_to_map_dict(footprint, original_map_data)
    with open(os.path.join(steps_dir, "map_step_000.json"), "w") as f:
        json.dump(init_map, f, indent=2)

    # ---- optimizer (only footprint centers) ----
    kw = dict(optimizer_kwargs or {})
    _name = optimizer_name.lower()
    if _name == "adam":
        opt = torch.optim.Adam([footprint.centers], lr=lr, **kw)
    elif _name == "adamw":
        opt = torch.optim.AdamW([footprint.centers], lr=lr, **kw)
    elif _name == "sgd":
        kw.setdefault("momentum", 0.9)
        opt = torch.optim.SGD([footprint.centers], lr=lr, **kw)
    elif _name == "rmsprop":
        opt = torch.optim.RMSprop([footprint.centers], lr=lr, **kw)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # ---- collect per-step trajectory data ----
    trajectory = []  # list of (N_trainable, 2) arrays in pixel coords

    def _centers_to_pixels(c):
        bx0, bx1, by0, by1 = [float(x) for x in footprint.bounds_xy]
        sx = (W - 1) / (bx1 - bx0 + 1e-12)
        sy = (H - 1) / (by1 - by0 + 1e-12)
        px = (c[:, 0] - bx0) * sx
        py = (c[:, 1] - by0) * sy
        return np.column_stack([px, py])

    # record initial positions
    with torch.no_grad():
        c0 = footprint.centers.detach().cpu().numpy()
        tmask_np = footprint.trainable_mask.cpu().numpy()
        trajectory.append(_centers_to_pixels(c0[tmask_np]))

    # ---- setup visualization ----
    with torch.no_grad():
        bmask_init = footprint.binary_mask().cpu().numpy()
        tmask_2d = np.zeros_like(bmask_init)
        # Mark trainable building pixels (approximate from centers)
        for idx in np.where(tmask_np)[0]:
            cx, cy = int(_centers_to_pixels(c0[idx:idx+1])[0, 0]), int(_centers_to_pixels(c0[idx:idx+1])[0, 1])
            r = 5  # approximate radius
            y0c, y1c = max(0, cy-r), min(H, cy+r)
            x0c, x1c = max(0, cx-r), min(W, cx+r)
            tmask_2d[y0c:y1c, x0c:x1c] = bmask_init[y0c:y1c, x0c:x1c]
        vis_logger.save_setup(bmask_init, tmask_2d, objective_rect, inlet_list)

    # ---- optimization loop ----
    for step in range(n_steps):
        opt.zero_grad()

        building_bin, fluid_bin = footprint()

        # combine fluid mask with objective region
        if obj_region is not None:
            obj_mask = fluid_bin * obj_region
        else:
            obj_mask = fluid_bin

        # predict for all inlets (rotating footprint so inlet is always from the left)
        u_pred_list, v_pred_list = [], []
        for k_idx in range(K):
            speed_k, k_rot = rotation_params[k_idx]
            if k_rot == 0:
                mask_k = building_bin[None, None]
            else:
                mask_k = torch.rot90(building_bin, k=k_rot, dims=(-2, -1))[None, None]

            u_k, v_k = solver(
                mask_k,
                torch.tensor([speed_k], device=device, dtype=torch.float32),
                torch.zeros(1, device=device, dtype=torch.float32),
            )

            if k_rot != 0:
                u_k, v_k = _rotate_velocity_back(u_k, v_k, k_rot)

            u_pred_list.append(u_k[0])  # remove batch dim
            v_pred_list.append(v_k[0])

        u_pred = torch.stack(u_pred_list)  # (K, T, H, W)
        v_pred = torch.stack(v_pred_list)

        # skip transient frames (eval only on post-transient)
        if transient_frames > 0:
            u_pred = u_pred[:, transient_frames:]
            v_pred = v_pred[:, transient_frames:]

        # --- objective ---
        # To use a custom objective, replace pwc_loss() below with your
        # function (see inverse/objective.py for the expected interface).
        loss_flow = 0.0
        metrics_accum = {}

        for k in range(K):
            _comfort_thr = (pwc_comfort_threshold[k]
                            if isinstance(pwc_comfort_threshold, (list, tuple))
                            else pwc_comfort_threshold)
            _stag_thr = (pwc_stagnation_threshold[k]
                         if isinstance(pwc_stagnation_threshold, (list, tuple))
                         else pwc_stagnation_threshold)
            out = pwc_loss(
                u=u_pred[k],
                v=v_pred[k],
                mask=obj_mask,
                tau=pwc_tau,
                danger_threshold=pwc_danger_threshold,
                comfort_threshold=_comfort_thr,
                stagnation_threshold=_stag_thr,
                w_danger=pwc_w_danger,
                w_comfort=pwc_w_comfort,
                w_stagnation=pwc_w_stagnation,
            )
            wk = inlet_weights[k]
            loss_flow = loss_flow + wk * out["total"]

            for key in out:
                if key == "total":
                    continue
                val = float(out[key].detach()) * float(wk)
                metrics_accum[key] = metrics_accum.get(key, 0.0) + val

        move_pen = footprint.movement_reg()
        cohesion_pen = footprint.cohesion_reg(cohesion_hinge=cohesion_hinge)
        loss_total = loss_flow + w_move * move_pen + w_cohesion * cohesion_pen

        loss_total.backward()
        opt.step()

        # ---- record metrics ----
        row = {
            "step": step,
            "loss_total": float(loss_total.detach()),
            "loss_flow": float(loss_flow.detach()),
            "move_pen": float(move_pen.detach()),
        }
        row.update(metrics_accum)
        csv_writer.writerow(row)
        csv_f.flush()

        # ---- visualization logger ----
        vis_logger.log(step, row)
        with torch.no_grad():
            # Average across inlets for visualization
            u_vis = u_pred.detach().mean(dim=0).cpu().numpy()  # [T, H, W]
            v_vis = v_pred.detach().mean(dim=0).cpu().numpy()
            bmask_step = building_bin.detach().cpu().numpy() > 0.5
        vis_logger.record_step(step, footprint, u_pred=u_vis, v_pred=v_vis,
                               bmask=bmask_step, obj_rect=objective_rect)

        # ---- record trajectory ----
        with torch.no_grad():
            cn = footprint.centers.detach().cpu().numpy()
            trajectory.append(_centers_to_pixels(cn[tmask_np]))

        # ---- save building JSON for this step ----
        with torch.no_grad():
            step_map = _footprint_to_map_dict(footprint, original_map_data)
        with open(os.path.join(steps_dir, f"map_step_{step+1:03d}.json"), "w") as f:
            json.dump(step_map, f, indent=2)

        # ---- print ----
        if (print_every is not None) and (step % int(print_every) == 0 or step == n_steps - 1):
            print(
                f"[{step:03d}] total={row['loss_total']:.6f} "
                f"flow={row['loss_flow']:.6f} mean={metrics_accum.get('mean_speed', 0):.3f} "
                f"danger={metrics_accum.get('danger', 0):.4f} "
                f"comfort={metrics_accum.get('comfort', 0):.4f} "
                f"stag={metrics_accum.get('stagnation', 0):.4f}"
                f" | run_dir={run_dir}"
            )

    csv_f.close()

    # ---- save trajectory as numpy ----
    traj_arr = np.stack(trajectory, axis=0)  # (n_steps+1, N_train, 2)
    np.save(os.path.join(run_dir, "trajectory.npy"), traj_arr)

    # ---- save final tensors ----
    torch.save(
        {
            "centers": footprint.centers.detach().cpu(),
            "centers0": footprint.centers0.detach().cpu(),
            "trainable_mask": footprint.trainable_mask.detach().cpu(),
            "sizes": footprint.sizes.detach().cpu(),
            "bounds_xy": footprint.bounds_xy.detach().cpu(),
            "parent_idx": footprint.parent_idx.detach().cpu(),
        },
        os.path.join(run_dir, "final_footprint_params.pt"),
    )

    # ---- final visualization ----
    with torch.no_grad():
        bmask_final = footprint.binary_mask().cpu().numpy()

        # Run solver one more time for initial and final flow fields
        # (initial uses bmask_init from before optimization)
        # We already have u_vis/v_vis from the last step for final flow
        u_final_np = u_vis
        v_final_np = v_vis

    # Get initial flow from first snapshot if available
    if 0 in vis_logger._flow_snapshots:
        u_init_np, v_init_np, _ = vis_logger._flow_snapshots[0]
    else:
        u_init_np, v_init_np = u_final_np, v_final_np

    vis_logger.save_final(
        footprint_module=footprint,
        bmask_initial=bmask_init,
        bmask_final=bmask_final,
        u_init=np.stack([u_init_np] * max(1, u_final_np.shape[0])) if u_init_np.ndim == 2 else np.expand_dims(u_init_np, 0).repeat(u_final_np.shape[0] if u_final_np.ndim == 3 else 1, axis=0),
        v_init=np.stack([v_init_np] * max(1, v_final_np.shape[0])) if v_init_np.ndim == 2 else np.expand_dims(v_init_np, 0).repeat(v_final_np.shape[0] if v_final_np.ndim == 3 else 1, axis=0),
        u_final=u_final_np if u_final_np.ndim == 3 else u_final_np[None],
        v_final=v_final_np if v_final_np.ndim == 3 else v_final_np[None],
        obj_rect=objective_rect,
        pwc_thresholds={
            "comfort_threshold": float(pwc_comfort_threshold[0]) if isinstance(pwc_comfort_threshold, (list, tuple)) else float(pwc_comfort_threshold),
            "danger_threshold": float(pwc_danger_threshold),
        },
    )
    vis_logger.close()

    return footprint, run_dir
