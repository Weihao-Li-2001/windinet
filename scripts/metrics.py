#!/usr/bin/env python3
"""
WinDiNet — Compute metrics from prediction .npz vs GT fields.npz.

Metrics: vRMSE, MAE (m/s), MRE (%), MSE, Spectral Divergence, Wasserstein (magnitudes).
All computed in m/s space on fluid cells only (building mask excluded).

Usage:
    python tools/metrics.py \
        --pred_dir /path/to/predictions \
        --samples_root /path/to/gt \
        --manifest dataset.json \
        --out_dir /path/to/metrics
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

MAG_CAP_MPS = 30.0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=Path, required=True)
    ap.add_argument("--samples_root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--num_samples", type=int, default=None)
    return ap.parse_args()


def _masked_mean(x, valid):
    return float(x[:, valid].mean()) if x.ndim == 3 else float(x[:, valid, :].mean())


def _masked_mre_pct(num, denom, valid, eps=1e-6):
    return float((np.abs(num[:, valid]) / np.maximum(np.abs(denom[:, valid]), eps)).mean() * 100.0)


@torch.no_grad()
def spectral_divergence(pred, target, mask, eps=1e-10):
    pred_f, targ_f = pred.float(), target.float()
    m = mask[:, 0, 0]
    m_c = m.unsqueeze(1).expand_as(pred_f[:, :, 0])
    fluid_pred = pred_f.permute(0, 1, 3, 4, 2)[m_c > 0.5]
    fluid_targ = targ_f.permute(0, 1, 3, 4, 2)[m_c > 0.5]
    if fluid_pred.shape[0] == 0:
        return 0.0
    P_pred = torch.fft.rfft(fluid_pred, dim=1).abs().pow(2)
    P_targ = torch.fft.rfft(fluid_targ, dim=1).abs().pow(2)
    return float((torch.log(P_pred + eps) - torch.log(P_targ + eps)).abs().mean(dim=1).mean().item())


@torch.no_grad()
def wasserstein_magnitudes(pred, target, mask):
    pred_f, targ_f = pred.float(), target.float()
    mag_pred = torch.sqrt(pred_f[:, 0] ** 2 + pred_f[:, 1] ** 2)
    mag_targ = torch.sqrt(targ_f[:, 0] ** 2 + targ_f[:, 1] ** 2)
    m = mask[:, 0, 0]
    fluid_pred = mag_pred.permute(0, 2, 3, 1)[m > 0.5]
    fluid_targ = mag_targ.permute(0, 2, 3, 1)[m > 0.5]
    if fluid_targ.shape[0] == 0:
        return 0.0
    return float((fluid_pred.sort(dim=1).values - fluid_targ.sort(dim=1).values).abs().mean(dim=1).mean().item())


def compute_metrics(pred_u, pred_v, gt_u, gt_v, valid):
    t = min(pred_u.shape[0], gt_u.shape[0])
    pu, pv = pred_u[:t].astype(np.float32), pred_v[:t].astype(np.float32)
    gu, gv = gt_u[:t].astype(np.float32), gt_v[:t].astype(np.float32)

    p_n = np.stack([pu, pv], axis=-1) / MAG_CAP_MPS * 0.5 + 0.5
    g_n = np.stack([gu, gv], axis=-1) / MAG_CAP_MPS * 0.5 + 0.5
    d_n = p_n - g_n

    mse = _masked_mean(d_n * d_n, valid)

    eps = 1e-12
    du_n, dv_n = d_n[..., 0], d_n[..., 1]
    mse_u = _masked_mean(du_n * du_n, valid)
    mse_v = _masked_mean(dv_n * dv_n, valid)
    var_u = float(np.var(g_n[..., 0][:, valid])) + eps
    var_v = float(np.var(g_n[..., 1][:, valid])) + eps
    vrmse = float(np.sqrt(0.5 * ((mse_u / var_u) + (mse_v / var_v))))

    g_mag_n = np.sqrt(g_n[..., 0] ** 2 + g_n[..., 1] ** 2)
    d_mag_n = np.sqrt(du_n ** 2 + dv_n ** 2)
    mre = _masked_mre_pct(d_mag_n, g_mag_n, valid)

    d_mps = np.stack([pu - gu, pv - gv], axis=-1)
    mae = _masked_mean(np.abs(d_mps), valid)

    F_len = pu.shape[0]
    pred_t = torch.from_numpy(np.stack([pu, pv], axis=0)[np.newaxis].copy())
    targ_t = torch.from_numpy(np.stack([gu, gv], axis=0)[np.newaxis].copy())
    mask_t = torch.from_numpy(valid.astype(np.float32))[None, None, None].expand(1, 1, F_len, -1, -1)

    return {
        "vrmse": vrmse,
        "mae": mae,
        "mre": mre,
        "mse": mse,
        "spectral_div": spectral_divergence(pred_t, targ_t, mask_t),
        "wasserstein_mag": wasserstein_magnitudes(pred_t, targ_t, mask_t),
    }


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    items = json.loads(args.manifest.read_text())
    if args.num_samples is not None:
        items = items[:args.num_samples]

    keys = ["vrmse", "mae", "mre", "mse", "spectral_div", "wasserstein_mag"]

    per_f = open(args.out_dir / "per_sample.csv", "w", newline="")
    per_w = csv.writer(per_f)
    per_w.writerow(["sample_id"] + keys)

    all_metrics, skipped = [], 0

    for i, it in enumerate(items):
        sid = Path(it["media_path"]).stem
        pred_path = args.pred_dir / f"{sid}.npz"
        if not pred_path.exists():
            skipped += 1
            continue

        pred = np.load(pred_path)
        gt = np.load(args.samples_root / sid / "fields.npz")

        pred_u, pred_v = pred["u_fields"], pred["v_fields"]
        valid = ~pred["bldg_mask"]
        gt_u, gt_v = gt["u_fields"], gt["v_fields"]

        m = compute_metrics(pred_u, pred_v, gt_u, gt_v, valid)
        all_metrics.append(m)
        per_w.writerow([sid] + [m[k] for k in keys])
        print(f"[{i+1}/{len(items)}] {sid} — vrmse={m['vrmse']:.4f}  mae={m['mae']:.3f}  W1={m['wasserstein_mag']:.3f}")

    per_f.close()

    if not all_metrics:
        print("No samples processed!")
        return

    summary = {}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        summary[f"mean_{k}"] = np.mean(vals)
        summary[f"std_{k}"] = np.std(vals)

    with open(args.out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "skipped"] + list(summary.keys()))
        w.writerow([len(all_metrics), skipped] + [summary[k] for k in summary.keys()])

    print(f"\n{len(all_metrics)} samples ({skipped} skipped)")
    print(f"  vrmse:          {summary['mean_vrmse']:.4f} +/- {summary['std_vrmse']:.4f}")
    print(f"  mae (m/s):      {summary['mean_mae']:.3f} +/- {summary['std_mae']:.3f}")
    print(f"  mse:            {summary['mean_mse']:.6f} +/- {summary['std_mse']:.6f}")
    print(f"  mre (%):        {summary['mean_mre']:.2f} +/- {summary['std_mre']:.2f}")
    print(f"  spectral_div:   {summary['mean_spectral_div']:.4f} +/- {summary['std_spectral_div']:.4f}")
    print(f"  wasserstein:    {summary['mean_wasserstein_mag']:.4f} +/- {summary['std_wasserstein_mag']:.4f}")


if __name__ == "__main__":
    main()
