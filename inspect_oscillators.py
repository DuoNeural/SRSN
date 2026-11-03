#!/usr/bin/env python3
"""
SRSN Oscillator Spectrum & Bifurcation Inspector
Analyzes omega_sq and alpha across all 12 layers x 320 dims (3,840 oscillators)
from saved checkpoints to track double-well potential bifurcations (alpha < 0, |alpha| > omega_sq)
connecting SDOC continuous physics to CDM crystallization dynamics.
"""

import os
import glob
import torch
import numpy as np

def analyze_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    opt_step = ckpt.get("opt_step", 0)
    loss = ckpt.get("loss", 0.0)
    state = ckpt["model_state"]

    alphas = []
    omegas = []
    layer_stats = []

    for l in range(12):
        a_key = f"layers.{l}.f_module.alpha"
        w_key = f"layers.{l}.f_module.omega_sq"
        if a_key in state and w_key in state:
            a = state[a_key].numpy()
            w = state[w_key].numpy()
            alphas.append(a)
            omegas.append(w)
            layer_stats.append({
                "layer": l,
                "a_min": float(a.min()),
                "a_max": float(a.max()),
                "a_mean": float(a.mean()),
                "a_neg": int(np.sum(a < 0)),
                "w_mean": float(w.mean()),
                "double_well": int(np.sum((a < 0) & (np.abs(a) > w)))
            })

    if not alphas:
        return None

    all_alphas = np.concatenate(alphas)
    all_omegas = np.concatenate(omegas)

    total_dims = len(all_alphas)
    neg_alphas = int(np.sum(all_alphas < 0))
    double_wells = int(np.sum((all_alphas < 0) & (np.abs(all_alphas) > all_omegas)))

    # Compute 5-bin histogram for alpha
    hist_counts, bin_edges = np.histogram(all_alphas, bins=5)
    hist_str = " | ".join([f"[{bin_edges[i]:.4f}, {bin_edges[i+1]:.4f}]: {hist_counts[i]}" for i in range(5)])

    res = {
        "opt_step": opt_step,
        "loss": loss,
        "total_dims": total_dims,
        "a_min": float(all_alphas.min()),
        "a_max": float(all_alphas.max()),
        "a_mean": float(all_alphas.mean()),
        "a_std": float(all_alphas.std()),
        "a_neg": neg_alphas,
        "double_wells": double_wells,
        "w_min": float(all_omegas.min()),
        "w_max": float(all_omegas.max()),
        "w_mean": float(all_omegas.mean()),
        "hist_str": hist_str,
        "layers": layer_stats
    }
    return res

def main():
    ckpt_dir = "/home/duoneural/srsn_ryzen/checkpoints"
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "srsn_step*.pt")))
    
    if not ckpts:
        print("[Inspector] No checkpoints found in", ckpt_dir)
        return

    print("=" * 80)
    print(f"[Inspector] Analyzing {len(ckpts)} Checkpoint(s) for Oscillator Spectrum Bifurcation")
    print("=" * 80)

    summary_rows = []
    for cp in ckpts:
        info = analyze_checkpoint(cp)
        if info:
            summary_rows.append(info)
            print(f"Step {info['opt_step']:04d} (Loss: {info['loss']:.4f}):")
            print(f"  Alpha:    min={info['a_min']:+.6f} | max={info['a_max']:+.6f} | mean={info['a_mean']:.6f} | std={info['a_std']:.6f}")
            print(f"  Negative: {info['a_neg']} / {info['total_dims']} ({info['a_neg']/info['total_dims']*100:.2f}%) | Double-Well (|alpha|>omega^2): {info['double_wells']}")
            print(f"  Omega_sq: min={info['w_min']:+.6f} | max={info['w_max']:+.6f} | mean={info['w_mean']:.6f}")
            print(f"  Hist:     {info['hist_str']}")
            print("-" * 80)

    # Write Markdown summary table to log
    log_path = "/home/duoneural/srsn_ryzen/oscillator_spectrum_log.md"
    with open(log_path, "w") as f:
        f.write("# DuoNeural SRSN Oscillator Spectrum & Bifurcation Log\n\n")
        f.write("| Step | Loss | Alpha Min | Alpha Max | Alpha Mean | Alpha Std | Alpha < 0 | Double-Well (|α|>ω²) | Omega² Mean |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in summary_rows:
            f.write(f"| {r['opt_step']} | {r['loss']:.4f} | {r['a_min']:+.5f} | {r['a_max']:+.5f} | {r['a_mean']:.5f} | {r['a_std']:.5f} | {r['a_neg']}/{r['total_dims']} | **{r['double_wells']}** | {r['w_mean']:.4f} |\n")
    print(f"\n[Inspector] Saved full spectrum markdown table to {log_path}")

if __name__ == "__main__":
    main()
