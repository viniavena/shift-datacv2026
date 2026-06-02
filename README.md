# SHIFT: Selecting High-Information Frames for Training Object Detectors on Driving Videos

> **Paper:** Vinicius Avena, Luan Pilon, Allan Pinto, Eduardo Valle  
> *Which Frames Matter? Frame Selection for Training Object Detectors on Driving Videos*  
> CVPR Workshop on Data-Centric Computer Vision (DataCV), 2026  
> [[Paper](https://openreview.net/)] <!-- replace with final DOI/arXiv link -->

> **Dataset:** AVADiP-DFS — Driving Frame Selection Benchmark  
> Hugging Face: [https://huggingface.co/datasets/](https://huggingface.co/datasets/) <!-- replace with final HF link -->  
> 160 driving video sequences · 48,000 frames at 30 FPS · exhaustive bounding-box annotations

---

## Overview

Training object detectors on video data is expensive: most frames are redundant because consecutive ones carry near-identical information.
**SHIFT** is a two-stage, training-free frame-selection algorithm that picks a compact, high-information subset for annotation and training given an annotation budget (in FPS).

### Stage 1 — Variation-Based Temporal Allocation

Inter-frame variation is measured via **Structural-Similarity Variation Density (SSVD)**:

```
FVI(t) = 1 − SSIM(I_t, I_{t−1})
```

Frames are allocated to temporal segments proportionally to their cumulative variation, so dynamic scenes receive more frames than static ones.

### Stage 2 — Greedy Diversity Maximisation (log-det)

From the Stage 1 pool (of size α · k, where α is an overselection factor), SHIFT greedily selects the final k frames by maximising the log-determinant of a cosine kernel matrix built on ResNet-50 embeddings:

```
S* = argmax_{|S|=k}  log det(L_S + ε I)
```

This is equivalent to a DPP-style diversity objective and is approximated in O(k · n) via a Cholesky rank-1 update.

---

## Repository Structure

```
.
├── methods/
│   ├── shift/
│   │   └── shift.py          # SHIFT algorithm (Stages 1 & 2, ablation modes)
│   └── baselines/
│       ├── afs.py            # Adaptive Frame Sampling (Yoon & Choi, CVPR 2023)
│       └── csod.py           # Coreset Selection for OD (Lee et al., CVPR 2024)
├── utils/
│   ├── sampling.py           # FrameInfo, video grouping, per-method wrappers
│   ├── fvi.py                # FVI variants: SSVD, OFVD, FSD, pixel-diff
│   ├── feature_extraction.py # ResNet-50 embedding & crop-feature extraction
│   ├── dataset.py            # YOLO dataset loading, subset copying, manifest
│   ├── detector.py           # run_detector_train / run_detector_val (YOLO backend)
│   └── results.py            # JSON result accumulation helpers
└── scripts/
    ├── run_shift.py          # SHIFT across annotation budgets (reproduces Table 2)
    ├── run_baselines.py      # All baselines across annotation budgets (Table 2)
    └── run_ablations.py      # SHIFT ablation study at 1 FPS (Table 3)
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.10 and a CUDA-capable GPU.

---

## Data Preparation

Download the **AVADiP-DFS** dataset from Hugging Face and unpack it.
The dataset must follow YOLO format:

```
avadip_dfs/
├── data.yaml          # path, train, val, test, nc, names
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Set the environment variable:

```bash
export DATASET_ROOT=/path/to/avadip_dfs
```

If your test split lives in a separate directory:

```bash
export TEST_DATASET_ROOT=/path/to/avadip_dfs_test
```

---

## Running Experiments

### SHIFT (our method)

```bash
# Default: evaluate budgets 0.3, 0.5, 1, 2, 3, 5, 10 FPS with SSVD FVI
python scripts/run_shift.py

# Custom budget list and FVI variant
SHIFT_BUDGETS=1,2,5 SHIFT_FVI_METHOD=ssvd python scripts/run_shift.py

# Label-aware variant (SHIFT-LA)
SHIFT_LABEL_AWARE=1 python scripts/run_shift.py
```

Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `DATASET_ROOT` | `/data/avadip_yolo` | Path to YOLO-format training dataset |
| `TEST_DATASET_ROOT` | `DATASET_ROOT` | Path to test split (may differ) |
| `SHIFT_BUDGETS` | `0.3,0.5,1,2,3,5,10` | Annotation budgets in FPS |
| `SHIFT_FVI_METHOD` | `ssvd` | FVI variant: `ssvd`, `ofvd`, `pixel_diff`, `fsd` |
| `SHIFT_OVERSELECT` | `3.0` | Overselection factor α for Stage 1 pool |
| `SHIFT_EMBEDDING_MODEL` | `resnet50` | Backbone for Stage 2 embeddings |
| `SHIFT_PCA_DIM` | `128` | PCA dimensionality before kernel computation |
| `SHIFT_LABEL_AWARE` | `0` | Set to `1` for SHIFT-LA |
| `YOLO_DEVICE` | `cuda` | Training device |
| `YOLO_EPOCHS` | `50` | Maximum training epochs |

Results are saved to `experiments_output/shift_runs/shift_results.json`.

---

### Baselines

```bash
# Run all baselines across all budgets
python scripts/run_baselines.py

# Run specific methods and budgets
EXPERIMENT_METHODS=ufs,rfs,afs_ssvd BUDGET_FPS=1,2,5 python scripts/run_baselines.py
```

Implemented baselines:

| ID | Method | Reference |
|---|---|---|
| `baseline_full` | Train on full dataset | — |
| `ufs` | Uniform Frame Sampling | — |
| `rfs` | Random Frame Sampling (5 seeds) | — |
| `afs_ssvd` | Adaptive FS — SSIM FVI | Yoon & Choi, CVPR 2023 |
| `afs_ofvd` | Adaptive FS — Optical Flow FVI | Yoon & Choi, CVPR 2023 |
| `afs_fsd` | Adaptive FS — Feature Similarity FVI | Yoon & Choi, CVPR 2023 |
| `csod` | Coreset Selection for OD | Lee et al., CVPR 2024 |

Results are saved to `experiments_output/baselines/baseline_results.json`.

---

### Ablation Study (Table 3)

```bash
python scripts/run_ablations.py
```

Ablations evaluated at 1 FPS:

| ID | Description |
|---|---|
| `ab_fvi_ofvd` | Stage 1 with optical-flow FVI instead of SSIM |
| `ab_fvi_pixdiff` | Stage 1 with pixel-difference FVI instead of SSIM |
| `ab_stage1_only` | Stage 1 only — no diversity maximisation |
| `ab_uniform_s2` | Stage 2 only — Stage 1 replaced by uniform sampling |

Results are saved to `experiments_output/shift_ablations_1fps/ablation_results.json`.

---

## Main Results (Table 2)

mAP@[0.5:0.95] on AVADiP-DFS test split. YOLO11X detector, 50 epochs.

| Method | 0.33 FPS | 0.5 FPS | 1 FPS | 2 FPS | 3 FPS | 5 FPS | 10 FPS |
|---|---|---|---|---|---|---|---|
| UFS | — | — | — | — | — | — | — |
| RFS | — | — | — | — | — | — | — |
| AFS-SSVD | — | — | — | — | — | — | — |
| AFS-OFVD | — | — | — | — | — | — | — |
| AFS-FSD | — | — | — | — | — | — | — |
| CSOD | — | — | — | — | — | — | — |
| **SHIFT (ours)** | — | — | — | — | — | — | — |
| Full dataset | — | — | — | — | — | — | — |

*Fill in numerical results from `shift_results.json` and `baseline_results.json` after running experiments.*

---

## Citation

If you use this code or the AVADiP-DFS dataset, please cite:

```bibtex
@inproceedings{avena2026shift,
  title     = {Which Frames Matter? Frame Selection for Training Object Detectors on Driving Videos},
  author    = {Avena, Vinicius and Pilon, Luan and Pinto, Allan and Valle, Eduardo},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (DataCV)},
  year      = {2026},
}
```

```bibtex
@dataset{avena2026avadip,
  title     = {{AVADiP-DFS}: Annotated Video Autonomous Driving in Peri-urban areas -- Driving Frame Selection Benchmark},
  author    = {Avena, Vinicius and Pilon, Luan and Pinto, Allan and Valle, Eduardo},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/},
}
```

---

## Acknowledgements

Baselines implemented based on:

- Y. Yoon and B. Choi, "Adaptive frame sampling for video object detection," *CVPR*, 2023.  
- H. Lee, S. Kim, J. Lee, J. Yoo, and N. Kwak, "Coreset selection for object detection," *CVPR*, 2024.

Detector training via [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).

---

## License

This code is released for academic research use only. See LICENSE for details.
