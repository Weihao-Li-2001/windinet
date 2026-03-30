"""
Shared visualization for wind field rendering.

All plots use a consistent style matching the published figures:
  - coolwarm colormap for wind speed (red=high, blue=low)
  - Semi-transparent dark grey building overlay
  - Green dashed rectangle for objective regions
  - Colorbars on the right with "Wind speed (m/s)" label

Functions accept numpy arrays — callers convert tensors before calling.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import imageio.v3 as iio

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

SPEED_CMAP = "coolwarm"
ERROR_CMAP = "hot"
BUILDING_RGBA = (0.3, 0.3, 0.3, 0.8)
FIXED_COLOR = (0.6, 0.6, 0.6)
TRAINABLE_COLOR = (0.2, 0.5, 0.9)
OBJ_RECT_COLOR = "#00CC00"
OBJ_RECT_LW = 3.0
FPS = 12
DPI_STATIC = 200
DPI_VIDEO = 150

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _speed(u, v):
    return np.sqrt(u.astype(np.float32) ** 2 + v.astype(np.float32) ** 2)


def _compute_vmax(speed, bmask):
    """99th percentile of fluid-pixel speeds."""
    fluid = speed[~bmask] if bmask.ndim == 2 else speed.reshape(-1)
    if fluid.size == 0:
        return 1.0
    return float(np.percentile(fluid, 99))


def _building_overlay(bmask):
    """Create RGBA overlay for buildings."""
    overlay = np.zeros((*bmask.shape, 4), dtype=np.float32)
    overlay[bmask, :3] = BUILDING_RGBA[0]
    overlay[bmask, 3] = BUILDING_RGBA[3]
    return overlay


# ---------------------------------------------------------------------------
# Matplotlib axes-level rendering
# ---------------------------------------------------------------------------

def render_speed_map(ax, u, v, bmask, *, vmax=None, label="Wind speed (m/s)", show_colorbar=True):
    """Plot wind speed magnitude on a matplotlib axis.

    Args:
        ax: matplotlib Axes.
        u, v: [H, W] velocity in m/s.
        bmask: [H, W] bool, True=building.
        vmax: colorbar max. Defaults to 99th percentile of fluid pixels.
        label: colorbar label.
        show_colorbar: whether to add a colorbar.

    Returns:
        AxesImage for the speed map.
    """
    speed = _speed(u, v)
    if vmax is None:
        vmax = _compute_vmax(speed, bmask)

    im = ax.imshow(speed, origin="lower", interpolation="nearest",
                   cmap=SPEED_CMAP, vmin=0, vmax=vmax)
    ax.imshow(_building_overlay(bmask), origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(label, fontsize=18)
        cb.ax.tick_params(labelsize=14)

    return im


def add_objective_rect(ax, rect):
    """Draw dashed green objective rectangle.

    Args:
        rect: [x0, x1, y0, y1] in pixel coordinates (matches config format).
    """
    x0, x1, y0, y1 = rect
    patch = Rectangle((x0, y0), x1 - x0, y1 - y0,
                       linewidth=OBJ_RECT_LW, edgecolor=OBJ_RECT_COLOR,
                       facecolor=(0, 0.8, 0, 0.08), linestyle="--")
    ax.add_patch(patch)


def render_building_map(ax, bmask, trainable_mask=None, obj_rect=None):
    """Plot building footprint with fixed/trainable coloring.

    Args:
        ax: matplotlib Axes.
        bmask: [H, W] bool, True=building.
        trainable_mask: [H, W] bool, True=trainable building pixels. If None, all grey.
        obj_rect: optional [x0, x1, y0, y1] for objective rectangle.
    """
    H, W = bmask.shape
    canvas = np.ones((H, W, 3), dtype=np.float32)

    if trainable_mask is not None:
        fixed = bmask & ~trainable_mask
        canvas[fixed] = FIXED_COLOR
        canvas[trainable_mask] = TRAINABLE_COLOR
    else:
        canvas[bmask] = FIXED_COLOR

    ax.imshow(canvas, origin="lower", interpolation="nearest")

    if obj_rect is not None:
        add_objective_rect(ax, obj_rect)

    # Legend
    handles = []
    from matplotlib.patches import Patch
    handles.append(Patch(facecolor=FIXED_COLOR, label="Fixed"))
    if trainable_mask is not None:
        handles.append(Patch(facecolor=TRAINABLE_COLOR, label="Optimisable"))
    if obj_rect is not None:
        handles.append(Patch(facecolor=(0, 0.8, 0, 0.15), edgecolor=OBJ_RECT_COLOR,
                             linestyle="--", linewidth=1.5, label="Objective region"))
    ax.legend(handles=handles, loc="upper left", fontsize=10, frameon=True,
              facecolor="white", edgecolor="lightgrey")

    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Frame rendering (numpy uint8 output)
# ---------------------------------------------------------------------------

def render_wind_frame(u, v, bmask, *, vmax=None):
    """Render a single wind speed frame as uint8 RGB.

    Args:
        u, v: [H, W] float m/s.
        bmask: [H, W] bool, True=building.
        vmax: max speed for colormap.

    Returns:
        [H, W, 3] uint8 array.
    """
    speed = _speed(u, v)
    if vmax is None:
        vmax = _compute_vmax(speed, bmask)

    cmap = matplotlib.colormaps[SPEED_CMAP]
    normed = np.clip(speed / max(vmax, 1e-8), 0, 1)
    rgb = (cmap(normed)[:, :, :3] * 255).astype(np.uint8)
    rgb[bmask] = [int(c * 255) for c in BUILDING_RGBA[:3]]
    return rgb


def render_error_frame(u_gt, v_gt, u_pred, v_pred, bmask, *, vmax=None):
    """Render GT | Prediction | Error as a single wide frame.

    Args:
        u_gt, v_gt, u_pred, v_pred: [H, W] float m/s.
        bmask: [H, W] bool.
        vmax: shared max for GT/Pred panels.

    Returns:
        [H, 3*W + 4, 3] uint8 array (2px white gaps between panels).
    """
    speed_gt = _speed(u_gt, v_gt)
    speed_pred = _speed(u_pred, v_pred)
    error = np.abs(speed_pred - speed_gt)

    if vmax is None:
        vmax = max(_compute_vmax(speed_gt, bmask), _compute_vmax(speed_pred, bmask))

    cmap_speed = matplotlib.colormaps[SPEED_CMAP]
    cmap_error = matplotlib.colormaps[ERROR_CMAP]

    gt_rgb = (cmap_speed(np.clip(speed_gt / max(vmax, 1e-8), 0, 1))[:, :, :3] * 255).astype(np.uint8)
    pred_rgb = (cmap_speed(np.clip(speed_pred / max(vmax, 1e-8), 0, 1))[:, :, :3] * 255).astype(np.uint8)

    emax = max(float(np.percentile(error[~bmask], 99)) if (~bmask).any() else 1.0, 1e-8)
    err_rgb = (cmap_error(np.clip(error / emax, 0, 1))[:, :, :3] * 255).astype(np.uint8)

    bldg_color = [int(c * 255) for c in BUILDING_RGBA[:3]]
    gt_rgb[bmask] = bldg_color
    pred_rgb[bmask] = bldg_color
    err_rgb[bmask] = bldg_color

    H, W = bmask.shape
    gap = np.full((H, 2, 3), 255, dtype=np.uint8)
    return np.concatenate([gt_rgb, gap, pred_rgb, gap, err_rgb], axis=1)


# ---------------------------------------------------------------------------
# Video rendering
# ---------------------------------------------------------------------------

def render_wind_video(u, v, bmask, path, *, fps=FPS, vmax=None):
    """Save wind magnitude video as MP4.

    Args:
        u, v: [T, H, W] float m/s.
        bmask: [H, W] bool.
        path: output file path.
        fps: frames per second.
        vmax: fixed max for colormap. If None, computed from all frames.
    """
    if vmax is None:
        all_speed = _speed(u, v)
        vmax = _compute_vmax(all_speed.reshape(-1, *bmask.shape)[-1], bmask)

    frames = [render_wind_frame(u[t], v[t], bmask, vmax=vmax) for t in range(u.shape[0])]
    iio.imwrite(str(path), np.stack(frames), fps=fps)


def render_error_video(u_gt, v_gt, u_pred, v_pred, bmask, path, *, fps=FPS, vmax=None):
    """Save GT | Prediction | Error video as MP4.

    Args:
        u_gt, v_gt, u_pred, v_pred: [T, H, W] float m/s.
        bmask: [H, W] bool.
        path: output file path.
    """
    if vmax is None:
        vmax = max(
            _compute_vmax(_speed(u_gt, v_gt).reshape(-1, *bmask.shape)[-1], bmask),
            _compute_vmax(_speed(u_pred, v_pred).reshape(-1, *bmask.shape)[-1], bmask),
        )

    T = u_gt.shape[0]
    frames = [render_error_frame(u_gt[t], v_gt[t], u_pred[t], v_pred[t], bmask, vmax=vmax)
              for t in range(T)]
    iio.imwrite(str(path), np.stack(frames), fps=fps)


def save_frame(frame, path):
    """Save a uint8 frame as PNG."""
    iio.imwrite(str(path), frame)
