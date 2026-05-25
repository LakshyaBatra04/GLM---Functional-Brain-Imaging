#!/usr/bin/env python
"""
group_analysis.py - Group Level Analysis with Path Auto-Fix
"""

import sys
import os
import re
import numpy as np
import nibabel as nib
from scipy.stats import t as t_dist, norm

def fix_path(path):
    """Converts /d/style/paths to D:/style/paths for Windows Python."""
    if re.match(r'^/[a-zA-Z]/', path):
        drive = path[1].upper()
        return drive + ":" + path[2:]
    return path

def t_to_z(t_values, df):
    z = np.zeros_like(t_values, dtype=np.float64)
    pos, neg = t_values > 0, t_values < 0
    if np.any(pos):
        p = np.clip(t_dist.sf(t_values[pos], df), 1e-300, 1 - 1e-15)
        z[pos] = norm.isf(p)
    if np.any(neg):
        p = np.clip(t_dist.cdf(t_values[neg], df), 1e-300, 1 - 1e-15)
        z[neg] = norm.ppf(p)
    return z

def main():
    if len(sys.argv) != 3:
        sys.exit(1)

    list_path, out_prefix = sys.argv[1], sys.argv[2]
    if os.path.dirname(out_prefix): os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    print("=" * 70)
    print("  Group Level Analysis (Path-Aware)")
    print("=" * 70)

    # Read and fix paths
    print("\n[1/5] Reading and fixing file list...")
    with open(list_path) as f:
        raw_files = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    
    files = [fix_path(f) for f in raw_files]
    N = len(files)
    print(f"  Subjects found: {N}")

    # Load data
    print("\n[2/5] Loading files...")
    first = nib.load(files[0])
    affine, header = first.affine, first.header
    shape3d = first.get_fdata().shape[:3]
    nv = np.prod(shape3d)

    data = np.zeros((N, nv), dtype=np.float64)
    for i, fpath in enumerate(files):
        if not os.path.exists(fpath):
            print(f"  FATAL ERROR: File not found: {fpath}")
            sys.exit(1)
        d = nib.load(fpath).get_fdata().astype(np.float64)
        data[i] = (d[:,:,:,0] if d.ndim == 4 else d).ravel()
        if i % 20 == 0 or i == N-1: print(f"    Progress: {i+1}/{N}")

    # Analysis
    print(f"\n[3/5] Running One-sample t-test (df={N-1})...")
    mean = np.mean(data, axis=0)
    se = np.std(data, axis=0, ddof=1) / np.sqrt(N)
    with np.errstate(divide='ignore', invalid='ignore'):
        tstat = np.where(se > 1e-15, mean / se, 0.0)

    print("[4/5] Converting t -> z...")
    zstat = t_to_z(tstat, N - 1)

    # Save
    print("[5/5] Saving outputs...")
    hdr = nib.Nifti1Header()
    hdr.set_data_shape(shape3d)
    hdr.set_zooms(list(header.get_zooms()[:3]))

    nib.save(nib.Nifti1Image(tstat.reshape(shape3d), affine, header=hdr), f"{out_prefix}.tstat.nii.gz")
    nib.save(nib.Nifti1Image(zstat.reshape(shape3d), affine, header=hdr), f"{out_prefix}.zstat.nii.gz")

    print("\n" + "=" * 70)
    print("  Group Analysis Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()