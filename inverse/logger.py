"""
Logging and visualization for inverse building layout optimization.

Produces publication-quality plots matching the style in the paper:
  - Setup plot: fixed/trainable buildings + objective region
  - Per-step: metric curves + mean flow snapshot + footprint snapshot
  - Final summary: trajectory, velocity stages, speed distribution
"""

import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, FancyArrowPatch

from windinet.visualization import (
    render_speed_map,
    render_building_map,
    render_wind_frame,
    render_wind_video,
    add_objective_rect,
    save_frame,
    SPEED_CMAP,
    BUILDING_RGBA,
    FIXED_COLOR,
    TRAINABLE_COLOR,
    OBJ_RECT_COLOR,
    DPI_STATIC,
)

plt.rcParams.update({"font.family": "serif", "font.size": 12})


class OptimizationLogger:
    """Logs metrics and saves publication-quality plots during optimization.

    Output directory structure::

        run_dir/
            config.json
            setup.png               # initial layout + objective region
            history.csv             # scalar metrics per step
            metrics.jsonl           # streaming JSON metrics
            frames/
                step_000000.png     # metric curves
            snapshots/
                footprint_step_000.png
                mean_flow_step_000.png
            steps/
                map_step_000.json   # building layout JSONs
            flow_initial.mp4
            flow_final.mp4
            trajectory_plot.png
            final_summary.png
            velocity_stages.png
            speed_distribution.png
    """

    def __init__(
        self,
        run_dir: str,
        plot_every: int = 1,
        write_csv_every: int = 25,
    ):
        self.run_dir = run_dir
        self.frames_dir = os.path.join(run_dir, "frames")
        self.snapshots_dir = os.path.join(run_dir, "snapshots")
        self.steps_dir = os.path.join(run_dir, "steps")
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(self.steps_dir, exist_ok=True)

        self.plot_every = int(plot_every)
        self.write_csv_every = int(write_csv_every)

        self.history = []
        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")
        self._metrics_f = open(self.metrics_path, "a", buffering=1)

        self.csv_path = os.path.join(run_dir, "history.csv")

        # Trajectory tracking
        self._trajectories = []  # list of (N_trainable, 2) pixel arrays
        self._flow_snapshots = {}  # step -> (u_mean, v_mean, bmask)

    def close(self):
        try:
            self._metrics_f.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Config + metrics logging
    # ------------------------------------------------------------------

    def write_config(self, config: dict):
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def log(self, step: int, metrics: dict):
        row = dict(metrics)
        row["step"] = int(step)
        self.history.append(row)
        self._metrics_f.write(json.dumps(row) + "\n")
        if step % self.write_csv_every == 0:
            self._write_csv()

    def _write_csv(self):
        if not self.history:
            return
        keys = sorted(k for k in self.history[-1] if isinstance(self.history[-1][k], (int, float)))
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for h in self.history:
                writer.writerow({k: h.get(k, "") for k in keys})

    # ------------------------------------------------------------------
    # Setup plot (beginning of optimization)
    # ------------------------------------------------------------------

    def save_setup(self, bmask, trainable_mask, obj_rect, inlets):
        """Save initial setup plot: building map + objective region + inlet arrows.

        Args:
            bmask: [H, W] bool, True=building.
            trainable_mask: [H, W] bool, True=trainable building pixels.
            obj_rect: [x0, y0, x1, y1] or None.
            inlets: list of [u, v] inlet velocities.
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        render_building_map(ax, bmask, trainable_mask, obj_rect)

        # Inlet arrows at left boundary
        H, W = bmask.shape
        for inlet in inlets:
            u_in, v_in = inlet
            speed = np.sqrt(u_in**2 + v_in**2)
            ax.annotate(
                f"{speed:.0f} m/s", xy=(15, H // 2), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="black", lw=2),
                xytext=(-30, H // 2), textcoords="data",
            )

        ax.set_title("Optimization setup", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, "setup.png"), dpi=DPI_STATIC)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Per-step visualization
    # ------------------------------------------------------------------

    def record_step(self, step, footprint_module, u_pred=None, v_pred=None, bmask=None, obj_rect=None):
        """Record trajectory and optionally save snapshots.

        Args:
            step: optimization step.
            footprint_module: DifferentiableFootprint with .centers, .trainable_mask, etc.
            u_pred, v_pred: [T, H, W] predicted velocity (for flow snapshots).
            bmask: [H, W] bool building mask.
            obj_rect: [x0, y0, x1, y1] or None.
        """
        # Always record trajectory
        self._record_trajectory(footprint_module)

        # Save flow snapshot for velocity stages plot
        if u_pred is not None and v_pred is not None:
            u_mean = u_pred.mean(axis=0) if u_pred.ndim == 3 else u_pred
            v_mean = v_pred.mean(axis=0) if v_pred.ndim == 3 else v_pred
            self._flow_snapshots[step] = (u_mean, v_mean, bmask)

        if step % self.plot_every != 0:
            return

        # Metric curves
        self._save_metric_plot(step)

        # Flow snapshot
        if u_pred is not None and bmask is not None:
            self._save_flow_snapshot(step, u_pred, v_pred, bmask, obj_rect)

        # Footprint snapshot
        self._save_footprint_snapshot(step, footprint_module)

    def _record_trajectory(self, footprint_module):
        with torch.no_grad():
            centers = footprint_module.centers.detach()
            bounds_xy = footprint_module.bounds_xy.detach()
            H, W = footprint_module.H, footprint_module.W
            pix = self._centers_to_pixels(centers, bounds_xy, H, W).cpu().numpy()
            train_mask = footprint_module.trainable_mask.cpu().numpy()
            self._trajectories.append(pix[train_mask].copy())

    @staticmethod
    def _centers_to_pixels(centers_xy, bounds_xy, H, W):
        bx0, bx1, by0, by1 = [float(x) for x in bounds_xy]
        x_pix = (centers_xy[:, 0] - bx0) / (bx1 - bx0 + 1e-12) * (W - 1)
        y_pix = (centers_xy[:, 1] - by0) / (by1 - by0 + 1e-12) * (H - 1)
        return torch.stack([x_pix, y_pix], dim=-1)

    def _save_metric_plot(self, step):
        """Save 3-panel metric curves: losses, PWC metrics, displacement."""
        steps = [h["step"] for h in self.history]
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # Loss curves
        axs[0].plot(steps, [h.get("loss_total", float("nan")) for h in self.history],
                    label="total", linewidth=1.5)
        axs[0].plot(steps, [h.get("loss_flow", float("nan")) for h in self.history],
                    label="flow", linewidth=1.0)
        axs[0].plot(steps, [h.get("move_pen", float("nan")) for h in self.history],
                    label="move_pen", linewidth=1.0)
        axs[0].set_title("Loss", fontsize=12)
        axs[0].set_xlabel("step")
        axs[0].legend(fontsize=9)
        axs[0].grid(True, alpha=0.3)

        # PWC metrics
        axs[1].plot(steps, [h.get("mean_speed", float("nan")) for h in self.history],
                    label="mean_speed", linewidth=1.5)
        axs[1].plot(steps, [h.get("danger", float("nan")) for h in self.history],
                    label="danger", linewidth=1.0)
        axs[1].plot(steps, [h.get("comfort", float("nan")) for h in self.history],
                    label="comfort", linewidth=1.0)
        axs[1].plot(steps, [h.get("stagnation", float("nan")) for h in self.history],
                    label="stagnation", linewidth=1.0)
        axs[1].set_title("PWC metrics", fontsize=12)
        axs[1].set_xlabel("step")
        axs[1].legend(fontsize=9)
        axs[1].grid(True, alpha=0.3)

        # Displacement
        if len(self._trajectories) > 1:
            traj = np.stack(self._trajectories, axis=0)
            origin = traj[0]
            disp = np.sqrt(((traj - origin[None]) ** 2).sum(axis=-1))
            for j in range(traj.shape[1]):
                axs[2].plot(range(len(disp)), disp[:, j],
                            color=plt.cm.tab10(j % 10), linewidth=0.8)
            axs[2].set_title("Building displacement (px)", fontsize=12)
            axs[2].set_xlabel("step")
            axs[2].grid(True, alpha=0.3)
        else:
            axs[2].set_visible(False)

        fig.tight_layout()
        fig.savefig(os.path.join(self.frames_dir, f"step_{step:06d}.png"), dpi=150)
        plt.close(fig)

    def _save_flow_snapshot(self, step, u_pred, v_pred, bmask, obj_rect):
        """Save time-averaged flow field snapshot."""
        u_mean = u_pred.mean(axis=0) if u_pred.ndim == 3 else u_pred
        v_mean = v_pred.mean(axis=0) if v_pred.ndim == 3 else v_pred

        fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))
        render_speed_map(ax, u_mean, v_mean, bmask)
        if obj_rect is not None:
            add_objective_rect(ax, obj_rect)
        ax.set_title(f"Mean flow — step {step}", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(self.snapshots_dir, f"mean_flow_step_{step:03d}.png"), dpi=DPI_STATIC)
        plt.close(fig)

    def _save_footprint_snapshot(self, step, footprint_module):
        """Save building footprint at this step."""
        with torch.no_grad():
            bprob = footprint_module.forward()[0].detach().cpu().numpy()

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(bprob, origin="lower", cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Footprint — step {step}", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(self.snapshots_dir, f"footprint_step_{step:03d}.png"), dpi=DPI_STATIC)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Final summary plots (end of optimization)
    # ------------------------------------------------------------------

    def save_final(self, footprint_module, bmask_initial, bmask_final,
                   u_init, v_init, u_final, v_final, obj_rect=None,
                   pwc_thresholds=None):
        """Generate all final summary plots.

        Args:
            footprint_module: final DifferentiableFootprint.
            bmask_initial, bmask_final: [H, W] bool building masks.
            u_init, v_init: [T, H, W] initial prediction.
            u_final, v_final: [T, H, W] final prediction.
            obj_rect: [x0, y0, x1, y1] or None.
            pwc_thresholds: dict with danger_threshold, comfort_threshold, stagnation_threshold.
        """
        run_dir = Path(self.run_dir)

        # Flow videos
        render_wind_video(u_init, v_init, bmask_initial, run_dir / "flow_initial.mp4")
        render_wind_video(u_final, v_final, bmask_final, run_dir / "flow_final.mp4")

        # Final summary: footprint + mean flow side by side
        self._save_final_summary(bmask_final, u_final, v_final, obj_rect)

        # Trajectory plot: initial vs final building positions
        self._save_trajectory_plot(footprint_module, obj_rect)

        # Velocity stages: row of snapshots at key steps
        self._save_velocity_stages(obj_rect)

        # Speed distribution: initial vs optimised histogram
        if pwc_thresholds is not None:
            self._save_speed_distribution(
                u_init, v_init, bmask_initial,
                u_final, v_final, bmask_final,
                obj_rect, pwc_thresholds,
            )

        # Final CSV
        self._write_csv()

    def _save_final_summary(self, bmask, u, v, obj_rect):
        """Binary footprint + mean flow side by side."""
        u_mean = u.mean(axis=0)
        v_mean = v.mean(axis=0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

        # Footprint
        canvas = np.ones((*bmask.shape, 3), dtype=np.float32)
        canvas[bmask] = 0.0
        ax1.imshow(canvas, origin="lower", interpolation="nearest")
        ax1.set_title("Optimised layout", fontsize=14)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Mean flow
        render_speed_map(ax2, u_mean, v_mean, bmask)
        if obj_rect is not None:
            add_objective_rect(ax2, obj_rect)
        ax2.set_title("Mean flow", fontsize=14)

        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, "final_summary.png"), dpi=DPI_STATIC)
        plt.close(fig)

    def _save_trajectory_plot(self, footprint_module, obj_rect):
        """Initial (blue dashed) vs final (red solid) building outlines."""
        if len(self._trajectories) < 2:
            return

        traj = np.stack(self._trajectories, axis=0)  # [T, N_train, 2]
        H = footprint_module.H
        W = footprint_module.W

        with torch.no_grad():
            centers = footprint_module.centers.detach()
            bounds_xy = footprint_module.bounds_xy.detach()
            all_pix = self._centers_to_pixels(centers, bounds_xy, H, W).cpu().numpy()
            train_mask = footprint_module.trainable_mask.cpu().numpy()
            fixed_pix = all_pix[~train_mask]

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")

        # Fixed buildings
        if len(fixed_pix) > 0:
            ax.scatter(fixed_pix[:, 0], fixed_pix[:, 1],
                       s=40, c=[FIXED_COLOR], marker="s", alpha=0.6, label="Fixed")

        # Initial positions (blue dashed)
        init_pos = traj[0]
        ax.scatter(init_pos[:, 0], init_pos[:, 1],
                   s=50, facecolors="none", edgecolors="blue", linewidths=1.5,
                   marker="s", label="Initial (trainable)", zorder=4)

        # Final positions (red solid)
        final_pos = traj[-1]
        ax.scatter(final_pos[:, 0], final_pos[:, 1],
                   s=50, c="red", marker="s", label="Final (trainable)", zorder=5)

        # Displacement arrows
        for j in range(init_pos.shape[0]):
            dx = final_pos[j, 0] - init_pos[j, 0]
            dy = final_pos[j, 1] - init_pos[j, 1]
            if np.sqrt(dx**2 + dy**2) > 1.0:
                ax.annotate("", xy=(final_pos[j, 0], final_pos[j, 1]),
                            xytext=(init_pos[j, 0], init_pos[j, 1]),
                            arrowprops=dict(arrowstyle="->", color="gray", lw=1.0, alpha=0.6))

        if obj_rect is not None:
            add_objective_rect(ax, obj_rect)

        ax.legend(loc="upper left", fontsize=9, frameon=True)
        ax.set_title("Building trajectory: initial (blue) → final (red)", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, "trajectory_plot.png"), dpi=DPI_STATIC)
        plt.close(fig)

    def _save_velocity_stages(self, obj_rect):
        """Row of mean flow snapshots at key optimization steps."""
        if not self._flow_snapshots:
            return

        all_steps = sorted(self._flow_snapshots.keys())
        if len(all_steps) < 2:
            return

        # Pick ~5 evenly spaced steps including first and last
        n_panels = min(5, len(all_steps))
        indices = np.linspace(0, len(all_steps) - 1, n_panels, dtype=int)
        selected = [all_steps[i] for i in indices]

        # Shared vmax across all panels
        vmax = 0
        for s in selected:
            u_m, v_m, bm = self._flow_snapshots[s]
            speed = np.sqrt(u_m**2 + v_m**2)
            fluid_speed = speed[~bm] if bm is not None else speed.ravel()
            if fluid_speed.size > 0:
                vmax = max(vmax, float(np.percentile(fluid_speed, 99)))

        fig, axs = plt.subplots(1, n_panels, figsize=(3.0 * n_panels, 5.5))
        if n_panels == 1:
            axs = [axs]

        for i, s in enumerate(selected):
            u_m, v_m, bm = self._flow_snapshots[s]
            label = "Initial" if i == 0 else (f"Step {s} (final)" if i == n_panels - 1 else f"Step {s}")
            render_speed_map(axs[i], u_m, v_m, bm, vmax=vmax,
                             show_colorbar=(i == n_panels - 1))
            if obj_rect is not None:
                add_objective_rect(axs[i], obj_rect)
            axs[i].set_title(label, fontsize=11)

        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, "velocity_stages.png"), dpi=DPI_STATIC)
        plt.close(fig)

    def _save_speed_distribution(self, u_init, v_init, bmask_init,
                                  u_final, v_final, bmask_final,
                                  obj_rect, thresholds):
        """Overlaid histogram: initial vs optimised speed distribution in objective region."""
        def _extract_speeds(u, v, bmask, rect):
            speed = np.sqrt(u.astype(np.float32)**2 + v.astype(np.float32)**2)
            if rect is not None:
                x0, y0, x1, y1 = [int(c) for c in rect]
                speed = speed[:, y0:y1, x0:x1]
                bmask = bmask[y0:y1, x0:x1]
            mask = ~bmask
            return speed[:, mask].ravel()

        speeds_init = _extract_speeds(u_init, v_init, bmask_init, obj_rect)
        speeds_final = _extract_speeds(u_final, v_final, bmask_final, obj_rect)

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        bins = np.linspace(0, max(speeds_init.max(), speeds_final.max(), 20), 80)
        ax.hist(speeds_init, bins=bins, density=True, alpha=0.4, color="steelblue", label="Initial")
        ax.hist(speeds_final, bins=bins, density=True, alpha=0.4, color="salmon", label="Optimised")
        ax.plot([], [], color="steelblue", linewidth=2)  # for legend line
        ax.plot([], [], color="salmon", linewidth=2)

        # Threshold lines
        comfort = thresholds.get("comfort_threshold", 5.0)
        danger = thresholds.get("danger_threshold", 15.0)
        ax.axvline(comfort, color="goldenrod", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.axvline(danger, color="crimson", linestyle="--", linewidth=1.5, alpha=0.8)

        ax.set_xlabel("Wind speed (m/s)", fontsize=14)
        ax.set_ylabel("Probability density", fontsize=14)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)

        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, "speed_distribution.png"), dpi=DPI_STATIC)
        plt.close(fig)
