#!/usr/bin/env python
"""
glm.py - Single Subject GLM Analysis for fMRI Data

Usage:
  python glm.py <functional_file> <ev_file_or_dir> <contrast_file> <output_prefix>
"""

import sys
import os
import re
import numpy as np
import nibabel as nib
from scipy.stats import t as t_dist, norm
from scipy.special import gamma as gamma_func


# ═══════════════════════════════════════════
# All 10 EVs in exact order
# ═══════════════════════════════════════════
EXPECTED_EV_FILES = [
    "EV1_audio_computation",
    "EV2_audio_left_hand",
    "EV3_audio_right_hand",
    "EV4_audio_sentence",
    "EV5_horizontal_checkerboard",
    "EV6_vertical_checkerboard",
    "EV7_video_computation",
    "EV8_video_left_hand",
    "EV9_video_right_hand",
    "EV10_video_sentence",
]

EV_LABELS = [
    "audio_computation",
    "audio_left_hand",
    "audio_right_hand",
    "audio_sentence",
    "horizontal_checkerboard",
    "vertical_checkerboard",
    "video_computation",
    "video_left_hand",
    "video_right_hand",
    "video_sentence",
]

CONTRAST_LABELS = [
    "horiz_checker - vert_checker",
    "checkerboard - video",
]


def load_functional_data(functional_file):
    """Load 4D fMRI NIfTI file."""
    print(f"  Loading: {functional_file}")
    img = nib.load(functional_file)
    data = img.get_fdata().astype(np.float64)
    header = img.header
    affine = img.affine

    tr = float(header.get_zooms()[3]) if len(header.get_zooms()) > 3 else 0.0
    if tr == 0 or tr is None:
        print("  WARNING: TR=0 in header. Defaulting to 2.4s.")
        tr = 2.4

    print(f"  Shape: {data.shape}")
    print(f"  TR: {tr}s")
    print(f"  Volumes: {data.shape[3]}")
    return img, data, affine, header, tr


def load_separate_ev_files(ev_directory):
    """Load 10 EV files from metadata directory."""
    print(f"  Directory: {ev_directory}")
    
    all_rows = []
    found_evs = []

    for idx, ev_filename in enumerate(EXPECTED_EV_FILES):
        condition_number = idx + 1
        full_path = os.path.join(ev_directory, ev_filename)

        if not os.path.exists(full_path):
            alt_path = full_path + ".txt"
            if os.path.exists(alt_path):
                full_path = alt_path
            else:
                print(f"    [MISSING] EV{condition_number:2d} ({EV_LABELS[idx]})")
                continue

        try:
            ev_content = np.loadtxt(full_path)
        except Exception as e:
            print(f"    [ERROR] EV{condition_number:2d}: {e}")
            continue

        if ev_content.ndim == 1:
            ev_content = ev_content.reshape(1, -1)

        n_events = ev_content.shape[0]
        condition_col = np.full((n_events, 1), condition_number)
        ev_with_cond = np.hstack([ev_content[:, :3], condition_col])
        all_rows.append(ev_with_cond)
        found_evs.append(condition_number)

        # Removed Unicode checkmark to prevent 'charmap' encoding errors
        print(f"    [OK] EV{condition_number:2d} ({EV_LABELS[idx]:30s}): {n_events:2d} events")

    if len(all_rows) == 0:
        print("\n  ERROR: No valid EV data loaded!")
        sys.exit(1)

    return np.vstack(all_rows), np.array(sorted(found_evs)), len(found_evs), \
           [EV_LABELS[c-1] for c in sorted(found_evs)]


def load_combined_ev_file(ev_file):
    ev_data = np.loadtxt(ev_file)
    if ev_data.ndim == 1:
        ev_data = ev_data.reshape(1, -1)
    condition_numbers = np.unique(ev_data[:, 3].astype(int))
    return ev_data, condition_numbers, len(condition_numbers), [f"EV{c}" for c in condition_numbers]


def load_contrast_file(contrast_file, n_conditions, ev_names):
    contrasts = np.loadtxt(contrast_file)
    if contrasts.ndim == 1:
        contrasts = contrasts.reshape(1, -1)
    return contrasts


def double_gamma_hrf(t):
    a1, b1, a2, b2, c = 6.0, 1.0, 16.0, 1.0, 1.0/6.0
    t = np.maximum(t, 0.0)
    peak = (t**(a1-1) * b1**a1 * np.exp(-b1*t)) / gamma_func(a1)
    undershoot = (t**(a2-1) * b2**a2 * np.exp(-b2*t)) / gamma_func(a2)
    return peak - c * undershoot


def create_design_matrix(ev_data, condition_numbers, n_conditions, n_volumes, tr):
    dt = 0.1
    total_time = n_volumes * tr
    n_highres = int(np.ceil(total_time / dt))
    hrf = double_gamma_hrf(np.arange(0, 32.0, dt))

    X_highres = np.zeros((n_highres, n_conditions))
    for i, cond_num in enumerate(condition_numbers):
        cond_events = ev_data[ev_data[:, 3].astype(int) == cond_num]
        stim = np.zeros(n_highres)
        for event in cond_events:
            s = max(0, int(np.round(event[0] / dt)))
            e = min(n_highres, int(np.round((event[0] + event[1]) / dt)))
            stim[s:e] = event[2]
        X_highres[:, i] = np.convolve(stim, hrf, mode='full')[:n_highres] * dt

    tr_idx = np.minimum(np.round(np.arange(n_volumes) * tr / dt).astype(int), n_highres - 1)
    return np.hstack([X_highres[tr_idx, :], np.ones((n_volumes, 1))]), X_highres[tr_idx, :]


def t_to_z(t_values, df):
    z = np.zeros_like(t_values, dtype=np.float64)
    pos, neg = t_values > 0, t_values < 0
    if np.any(pos):
        z[pos] = norm.isf(np.clip(t_dist.sf(t_values[pos], df), 1e-300, 1.0 - 1e-15))
    if np.any(neg):
        z[neg] = norm.ppf(np.clip(t_dist.cdf(t_values[neg], df), 1e-300, 1.0 - 1e-15))
    return z


def save_nifti(data_3d, affine, header, filename):
    hdr = nib.Nifti1Header()
    hdr.set_data_shape(data_3d.shape)
    hdr.set_zooms(list(header.get_zooms()[:3]))
    nib.save(nib.Nifti1Image(data_3d.astype(np.float64), affine, header=hdr), filename)


def run_glm(data, X_full, n_conditions, contrasts, affine, header, output_prefix, ev_names):
    nx, ny, nz, T = data.shape
    p = X_full.shape[1]
    df = T - p
    XtX_inv = np.linalg.pinv(X_full.T @ X_full)
    pinvX = XtX_inv @ X_full.T
    
    data_2d = data.reshape(-1, T).T
    beta_hat = pinvX @ data_2d
    residuals = data_2d - X_full @ beta_hat
    sigma2 = np.sum(residuals**2, axis=0) / df

    # Save PEs
    for i in range(n_conditions):
        save_nifti(beta_hat[i, :].reshape(nx, ny, nz), affine, header, f"{output_prefix}.pe{i+1}.nii.gz")

    # Save Contrasts
    contrasts_full = np.zeros((contrasts.shape[0], p))
    contrasts_full[:, :n_conditions] = contrasts

    for j in range(contrasts.shape[0]):
        c = contrasts_full[j, :]
        cope = c @ beta_hat
        se = np.sqrt(np.maximum(sigma2 * (c @ XtX_inv @ c), 0))
        tstat = np.where(se > 1e-15, cope / se, 0.0)
        zstat = t_to_z(tstat, df)

        save_nifti(cope.reshape(nx, ny, nz), affine, header, f"{output_prefix}.cope{j+1}.nii.gz")
        save_nifti(tstat.reshape(nx, ny, nz), affine, header, f"{output_prefix}.tstat{j+1}.nii.gz")
        save_nifti(zstat.reshape(nx, ny, nz), affine, header, f"{output_prefix}.zstat{j+1}.nii.gz")
        print(f"    [DONE] Contrast {j+1}: {CONTRAST_LABELS[j] if j < 2 else j+1}")


def main():
    if len(sys.argv) != 5:
        sys.exit(1)

    func_file, ev_path, con_file, out_prefix = sys.argv[1:5]
    if os.path.dirname(out_prefix): os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    img, data, affine, header, tr = load_functional_data(func_file)
    if os.path.isdir(ev_path):
        ev_data, cond_nums, n_cond, ev_names = load_separate_ev_files(ev_path)
    else:
        ev_data, cond_nums, n_cond, ev_names = load_combined_ev_file(ev_path)

    contrasts = load_contrast_file(con_file, n_cond, ev_names)
    X_full, X_ev = create_design_matrix(ev_data, cond_nums, n_cond, data.shape[3], tr)
    run_glm(data, X_full, n_cond, contrasts, affine, header, out_prefix, ev_names)


if __name__ == "__main__":
    main()