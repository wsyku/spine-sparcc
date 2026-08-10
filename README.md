# SpineSPARCC

**SpineSPARCC** is an inference-only research pipeline for sagittal
fat-suppressed spine MRI. It performs BME segmentation with a five-fold
nnU-Net ensemble, slice- and segment-level BME classification, segment-level
SPARCC score regression, and patient-level aggregation.

Patient images, training data, dataset splits, optimizer states, training
logs, and historical experiments are not distributed.

> This software is for research use only. It is not a medical device and must
> not be used as the sole basis for diagnosis or treatment decisions.

## Repository layout

```text
SpineSPARCC/
├── config/inference.yaml
├── examples/
│   ├── input/
│   └── output/
├── models/
│   ├── classification/
│   ├── regression/
│   └── segmentation/
├── src/
├── tools/
├── MODEL_CARD.md
├── THIRD_PARTY_NOTICES.md
└── requirements.txt
```

## Input

Place NIfTI images in one directory using these names:

```text
<patient_id>_C_0000.nii.gz
<patient_id>_T_0000.nii.gz
<patient_id>_L_0000.nii.gz
```

`C`, `T`, and `L` denote cervical, thoracic, and lumbar segments. A patient
may have one, two, or all three segment files.

## Installation

The verified reference environment is Python 3.12.13, PyTorch 2.12.1,
torchvision 0.27.1, CUDA 13.2, nnU-Net v2.8.0, Windows 11, and an NVIDIA RTX
5090 GPU.

Clone the repository and install its pinned dependencies:

```powershell
git clone <repository-url>
cd spine-sparcc
python -m pip install -r requirements.txt
```

This repository is not installed as a Python package. For a GPU installation,
install the PyTorch build appropriate for the local CUDA setup if it differs
from the reference environment. CPU inference is supported but substantially
slower.

## Model weights

Download `spine-sparcc-models-v0.1.0.zip` from the project's GitHub Release and
extract its contents directly into `models/`. The resulting layout and
checksum verification command are documented in `models/README.md`.

## Run inference

The repository does not include patient images. After adding locally
authorized images to `examples/input`, run from the repository root:

```powershell
python tools/verify_release.py
python src/run_inference.py --input examples/input --output examples/output
```

The inference command produces:

```text
examples/output/
├── segmentation/
├── slice_classification.csv
├── segment_results.csv
└── patient_summary.csv
```

`slice_classification.csv` contains original/model slice indices, raw slice
probabilities, and thresholded slice labels. `segment_results.csv` contains
raw segment probabilities, thresholded segment labels, raw regression scores,
and classification-gated SPARCC scores. Regression scores are bounded to the
valid 0-108 range.

The segment threshold is 0.6978082060813904, selected on the internal
validation set and fixed before external testing. A classifier-negative
segment receives a final SPARCC score of zero; `sparcc_score_raw` is retained
for transparent downstream analysis.

## Reusing segmentation outputs

Retain nnU-Net probability arrays during the first run:

```powershell
python src/run_inference.py --input examples/input --output examples/output --save-probabilities
```

After the masks and probability arrays exist, reuse them with:

```powershell
python src/run_inference.py --input examples/input --output examples/output --skip-segmentation --save-probabilities
```

## Reproducibility notes

- Inference seed: 42.
- cuDNN benchmarking is disabled and deterministic mode is enabled.
- Volumes use the central 16 slices; shorter volumes are padded at the end.
- Padded slices and invalid adjacent pairs are excluded.
- MRI and probability maps are resized to 320 × 320 with bilinear interpolation.
- MRI foreground voxels (`intensity > 0`) are z-score normalized.
- Only segmentation is a five-fold ensemble. Classification and regression
  each use one selected fold-0 checkpoint.

Small floating-point differences can occur across GPU models, CUDA versions,
and PyTorch builds. Classification decisions should be compared using the
released thresholds; continuous outputs should use a numerical tolerance.

## Citation and license

See `CITATION.cff` for citation metadata, `LICENSE` for the code license, and
`THIRD_PARTY_NOTICES.md` for third-party components. Model weights are
distributed under the terms stated in `MODEL_CARD.md`.
