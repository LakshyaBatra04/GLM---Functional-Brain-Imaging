# fMRI GLM Analysis Pipeline

A lightweight, dependency-minimal Python pipeline for **first-level (single-subject) and second-level (group) fMRI GLM analysis**, without relying on FSL, SPM, or Nilearn. Implements the full analysis chain from raw 4D BOLD data to z-statistic maps.

---

## Overview

This pipeline provides two standalone scripts:

| Script | Purpose |
|---|---|
| `glm.py` | Single-subject GLM — design matrix construction, parameter estimation, contrast computation |
| `group_analysis.py` | Group-level one-sample t-test across subjects' contrast maps |

---

## Features

- **Double-gamma HRF convolution** at 0.1s resolution, downsampled to TR
- Supports **10 experimental conditions** (audio/video × computation/hand/sentence/checkerboard)
- Accepts EV files as either a **combined file** (onset, duration, amplitude, condition) or a **directory of separate per-condition files**
- Outputs **parameter estimates (PEs), COPEs, t-stats, and z-stats** as NIfTI files
- **Numerically stable t→z conversion** using scipy's survival/CDF functions with clamped precision
- Group analysis with automatic **Windows path fixing** (`/d/path/` → `D:/path/`)

---

## Requirements

```
numpy
nibabel
scipy
```

Install with:
```bash
pip install numpy nibabel scipy
```

---

## Usage

### Single-Subject GLM

```bash
python glm.py <functional_file> <ev_file_or_dir> <contrast_file> <output_prefix>
```

**Arguments:**

| Argument | Description |
|---|---|
| `functional_file` | 4D BOLD NIfTI file (`.nii` or `.nii.gz`) |
| `ev_file_or_dir` | Combined EV file **or** directory containing separate EV files |
| `contrast_file` | Text file with one contrast vector per line |
| `output_prefix` | Prefix for all output files (directories created automatically) |

**Example:**
```bash
python glm.py sub-01_bold.nii.gz evs/ contrasts.txt output/sub-01
```

**Outputs:**
```
output/sub-01.pe1.nii.gz   ... sub-01.pe10.nii.gz    # Parameter estimates
output/sub-01.cope1.nii.gz ... sub-01.cope2.nii.gz   # Contrast estimates
output/sub-01.tstat1.nii.gz                           # T-statistic maps
output/sub-01.zstat1.nii.gz                           # Z-statistic maps
```

---

### Group Analysis

```bash
python group_analysis.py <file_list> <output_prefix>
```

**Arguments:**

| Argument | Description |
|---|---|
| `file_list` | Text file with one subject COPE path per line (lines starting with `#` are ignored) |
| `output_prefix` | Prefix for output files |

**Example:**
```bash
python group_analysis.py cope1_paths.txt output/group_cope1
```

**Outputs:**
```
output/group_cope1.tstat.nii.gz
output/group_cope1.zstat.nii.gz
```

---

## File Formats

### EV File (combined)

Four columns: `onset  duration  amplitude  condition_number`

```
0.0    15.0    1.0    1
30.0   15.0    1.0    2
60.0   15.0    1.0    1
90.0   15.0    1.0    3
```

### EV Directory (separate files)

Named exactly as:
```
EV1_audio_computation
EV2_audio_left_hand
EV3_audio_right_hand
EV4_audio_sentence
EV5_horizontal_checkerboard
EV6_vertical_checkerboard
EV7_video_computation
EV8_video_left_hand
EV9_video_right_hand
EV10_video_sentence
```

Each file has three columns: `onset  duration  amplitude`

### Contrast File

One contrast vector per line, with one weight per condition (10 values for 10 EVs):

```
0 0 0 0 1 -1 0 0 0 0     # horiz_checkerboard - vert_checkerboard
0 0 0 0 1 1 -1 -1 -1 -1  # checkerboard - video
```

---

## Pipeline Details

### HRF Model

Double-gamma HRF (SPM-style):

```
h(t) = t^(a1-1) * e^(-b1*t) / Γ(a1)  -  (1/6) * t^(a2-1) * e^(-b2*t) / Γ(a2)
```

with `a1=6, b1=1, a2=16, b2=1`. Convolution is performed at 0.1s resolution and downsampled to TR.

### GLM

Standard OLS: `β̂ = (XᵀX)⁻¹Xᵀy`, with the intercept appended as the last column of the design matrix. Residual variance is computed per-voxel with `df = T - p`.

### Contrast Estimation

For contrast vector `c`:
- `COPE = cᵀβ̂`
- `SE = sqrt(σ² · cᵀ(XᵀX)⁻¹c)`
- `t = COPE / SE`
- `z = Φ⁻¹(F_t(t; df))` (numerically stable conversion)

---

## Notes

- If TR is missing or zero in the NIfTI header, it defaults to **2.4s**
- The group analysis script handles Git-Bash/MSYS2-style Unix paths on Windows automatically
- Output directories are created automatically if they don't exist

---

## Experimental Conditions

This pipeline was designed for a dataset with 10 conditions across audio/video modalities:

| EV | Condition |
|---|---|
| EV1 | Audio — Computation |
| EV2 | Audio — Left Hand |
| EV3 | Audio — Right Hand |
| EV4 | Audio — Sentence |
| EV5 | Horizontal Checkerboard |
| EV6 | Vertical Checkerboard |
| EV7 | Video — Computation |
| EV8 | Video — Left Hand |
| EV9 | Video — Right Hand |
| EV10 | Video — Sentence |
