"""
Building footprint parametrization for inverse optimization.

The default parametrization (``DifferentiableFootprint``) represents buildings
as axis-aligned rectangles with trainable centre positions.

Extending with custom parametrizations
---------------------------------------
To use a different shape representation (e.g. splines, polygons, or
level-set fields), create a new ``nn.Module`` that satisfies this interface::

    class MyFootprint(nn.Module):
        def __init__(self, json_path, H, W, device="cpu", **kwargs):
            super().__init__()
            self.H, self.W = H, W
            # ... load initial layout, create trainable parameters ...

        def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
            '''Return (building_prob, fluid_prob), each (H, W) in [0, 1].
            Must be differentiable w.r.t. trainable parameters.'''
            ...

        def binary_mask(self, threshold=0.5) -> torch.Tensor:
            '''Return (H, W) bool mask for visualization.'''
            ...

        def movement_reg(self) -> torch.Tensor:
            '''Optional regularization loss (return 0 if unused).'''
            ...

        def cohesion_reg(self, cohesion_hinge=0.0) -> torch.Tensor:
            '''Optional cohesion loss (return 0 if unused).'''
            ...

    The optimizer in ``inverse/optimize.py`` calls ``footprint()`` to get the
    soft mask, passes it to the surrogate model, and calls ``.movement_reg()``
    and ``.cohesion_reg()`` for regularization. Replace
    ``DifferentiableFootprint`` with your class in the optimization script.
"""

import json
import torch
import torch.nn as nn


class DifferentiableFootprint(nn.Module):
    """Soft building occupancy from axis-aligned rectangles.

    Each building is a soft sigmoid rectangle. Trainable buildings optimise
    their centre (cx, cy); width and height remain fixed.

    When ``subdivide > 1``, each trainable building is split into a
    subdivide x subdivide grid of independently movable sub-blocks.
    """
    def __init__(self, json_path, H, W, tau=2.0, device="cpu", subdivide=1):
        super().__init__()
        self.H, self.W = H, W
        self.tau = float(tau)
        self.subdivide = int(subdivide)

        data = json.load(open(json_path, "r"))
        bx0, bx1, by0, by1, *_ = data["bounds"]["extents"]

        xs = torch.linspace(bx0, bx1, W, device=device)
        ys = torch.linspace(by0, by1, H, device=device)
        Y, X = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("X", X)  # (H,W)
        self.register_buffer("Y", Y)  # (H,W)

        centers, sizes, trainable, parent = [], [], [], []
        for i, b in enumerate(data["blocks"]):
            xmin, xmax, ymin, ymax, *_ = b["extents"]
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            w = (xmax - xmin)
            h = (ymax - ymin)
            is_train = bool(b.get("trainable", False))

            if is_train and self.subdivide > 1:
                # split into subdivide x subdivide sub-blocks
                sub_w = w / self.subdivide
                sub_h = h / self.subdivide
                for row in range(self.subdivide):
                    for col in range(self.subdivide):
                        sub_cx = xmin + sub_w * (col + 0.5)
                        sub_cy = ymin + sub_h * (row + 0.5)
                        centers.append([sub_cx, sub_cy])
                        sizes.append([sub_w, sub_h])
                        trainable.append(True)
                        parent.append(i)
            else:
                centers.append([cx, cy])
                sizes.append([w, h])
                trainable.append(is_train)
                parent.append(i)

        self.centers = nn.Parameter(torch.tensor(centers, dtype=torch.float32, device=device))  # (N,2)
        self.register_buffer("sizes", torch.tensor(sizes, dtype=torch.float32, device=device))  # (N,2)
        self.register_buffer("trainable_mask", torch.tensor(trainable, dtype=torch.bool, device=device))  # (N,)
        self.register_buffer("parent_idx", torch.tensor(parent, dtype=torch.long, device=device))  # (N,)
        self.register_buffer("bounds_xy", torch.tensor([bx0, bx1, by0, by1], device=device))

        # Save original centers for optional movement regularization
        self.register_buffer("centers0", self.centers.detach().clone())

    def forward(self):
        """Compute soft (differentiable) building and fluid probability maps.

        Returns:
            (building_prob, fluid_prob): each (H, W) in [0, 1].
        """
        # freeze non-trainable building centers
        centers = torch.where(self.trainable_mask[:, None], self.centers, self.centers.detach())

        w, h = self.sizes[:, 0], self.sizes[:, 1]
        bx0, bx1, by0, by1 = self.bounds_xy

        # keep rectangles inside bounds (hard clamp)
        cx = centers[:, 0].clamp(bx0 + 0.5*w, bx1 - 0.5*w)
        cy = centers[:, 1].clamp(by0 + 0.5*h, by1 - 0.5*h)

        xmin = cx - 0.5*w
        xmax = cx + 0.5*w
        ymin = cy - 0.5*h
        ymax = cy + 0.5*h

        X, Y = self.X, self.Y
        tau = self.tau
        sig = torch.sigmoid

        px = sig((X[None] - xmin[:, None, None]) / tau) * sig((xmax[:, None, None] - X[None]) / tau)
        py = sig((Y[None] - ymin[:, None, None]) / tau) * sig((ymax[:, None, None] - Y[None]) / tau)
        p_all = (px * py).clamp(0.0, 1.0)  # (N,H,W)

        # soft union
        building_prob = 1.0 - torch.prod(1.0 - p_all, dim=0)  # (H,W)
        fluid_prob = 1.0 - building_prob
        return building_prob, fluid_prob

    def binary_mask(self, threshold: float = 0.5) -> torch.Tensor:
        """Return a hard binary building mask (for visualization only, not differentiable).

        Returns:
            (H, W) bool tensor, True = building.
        """
        with torch.no_grad():
            building_prob, _ = self.forward()
            return building_prob > threshold

    def focus_mask(self, radius_px: float) -> torch.Tensor:
        """Return a soft (H,W) mask that is ~1 within `radius_px` pixels of
        any trainable building centre, smoothly falling to 0 beyond that.

        Works in pixel space so the radius is independent of world units.
        """
        bx0, bx1, by0, by1 = self.bounds_xy
        # world-to-pixel scale factors
        sx = (self.W - 1) / (bx1 - bx0 + 1e-12)
        sy = (self.H - 1) / (by1 - by0 + 1e-12)

        centers = self.centers.detach()[self.trainable_mask]  # (M, 2)
        # convert to pixel coords
        cx_px = (centers[:, 0] - bx0) * sx  # (M,)
        cy_px = (centers[:, 1] - by0) * sy  # (M,)

        # pixel grid
        gx = torch.arange(self.W, device=self.X.device, dtype=torch.float32)
        gy = torch.arange(self.H, device=self.X.device, dtype=torch.float32)
        GY, GX = torch.meshgrid(gy, gx, indexing="ij")  # (H, W)

        # distance from each pixel to nearest trainable centre
        dx = GX[None] - cx_px[:, None, None]  # (M, H, W)
        dy = GY[None] - cy_px[:, None, None]
        dist = torch.sqrt(dx ** 2 + dy ** 2)      # (M, H, W)
        min_dist = dist.min(dim=0).values          # (H, W)

        # smooth falloff: 1 inside radius, sigmoid decay outside
        return torch.sigmoid((radius_px - min_dist) / (radius_px * 0.15 + 1e-6))

    def downstream_mask(
        self, width_px: float = 50.0, pad_px: float = 20.0, gap_px: float = 5.0,
    ) -> torch.Tensor:
        """Fixed rectangular mask *downstream* (to the right) of trainable buildings.

        Computed from **initial** positions (``centers0``), so it does NOT move
        during optimisation.

        Parameters
        ----------
        width_px : float
            Horizontal extent of the measurement rectangle (pixels).
        pad_px : float
            Vertical padding above/below the building cluster (pixels).
        gap_px : float
            Horizontal gap between rightmost building edge and the start of
            the measurement region (pixels).
        """
        bx0, bx1, by0, by1 = self.bounds_xy
        sx = (self.W - 1) / (bx1 - bx0 + 1e-12)
        sy = (self.H - 1) / (by1 - by0 + 1e-12)

        mask = self.trainable_mask
        c0 = self.centers0[mask]   # (M, 2)  — initial centres
        s0 = self.sizes[mask]      # (M, 2)

        # building edges in world coords
        right_edges = c0[:, 0] + s0[:, 0] / 2
        top_edges   = c0[:, 1] + s0[:, 1] / 2
        bot_edges   = c0[:, 1] - s0[:, 1] / 2

        # bounding box → pixel coords
        x_start_px = float((right_edges.max() - bx0) * sx) + gap_px
        x_end_px   = x_start_px + width_px
        y_min_px   = float((bot_edges.min() - by0) * sy) - pad_px
        y_max_px   = float((top_edges.max() - by0) * sy) + pad_px

        # clamp to image
        x_end_px = min(x_end_px, float(self.W - 1))
        y_min_px = max(y_min_px, 0.0)
        y_max_px = min(y_max_px, float(self.H - 1))

        # pixel grid
        gx = torch.arange(self.W, device=self.X.device, dtype=torch.float32)
        gy = torch.arange(self.H, device=self.X.device, dtype=torch.float32)
        GY, GX = torch.meshgrid(gy, gx, indexing="ij")

        # soft sigmoid edges (sharp-ish, ~3 px transition)
        edge = 2.0
        mask_x = (torch.sigmoid((GX - x_start_px) / edge)
                  * torch.sigmoid((x_end_px - GX) / edge))
        mask_y = (torch.sigmoid((GY - y_min_px) / edge)
                  * torch.sigmoid((y_max_px - GY) / edge))

        return mask_x * mask_y

    def movement_reg(self):
        """Optional: penalize moving trainable centers too far from initial."""
        d = self.centers - self.centers0
        d = d[self.trainable_mask]  # only trainable
        return (d**2).mean() if d.numel() else torch.tensor(0.0, device=self.centers.device)

    def cohesion_reg(self, cohesion_hinge: float = 0.0):
        """Penalize sub-blocks of the same parent drifting apart.

        For each parent building, computes the spread of sub-block
        displacements relative to their mean displacement.

        If ``cohesion_hinge > 0``, uses a hinge formulation: sub-blocks are
        free to deviate up to ``cohesion_hinge`` world-units from the mean
        displacement; only excess spread is penalized.  This prevents the
        penalty from dominating when the group translates as a whole.

        Zero when subdivide <= 1.
        """
        if self.subdivide <= 1:
            return torch.tensor(0.0, device=self.centers.device)

        # displacement from initial position for trainable blocks
        mask = self.trainable_mask
        disp = (self.centers - self.centers0)[mask]  # (M, 2)
        parents = self.parent_idx[mask]               # (M,)

        unique_parents = parents.unique()
        loss = torch.tensor(0.0, device=self.centers.device)
        count = 0
        for pid in unique_parents:
            sel = parents == pid
            if sel.sum() <= 1:
                continue
            d = disp[sel]  # (K, 2)
            mean_d = d.mean(0, keepdim=True)
            dev = torch.norm(d - mean_d, dim=-1)  # (K,) L2 deviation per sub-block
            if cohesion_hinge > 0.0:
                loss = loss + (torch.clamp(dev - cohesion_hinge, min=0.0) ** 2).mean()
            else:
                loss = loss + (dev ** 2).mean()
            count += 1

        return loss / max(count, 1)
