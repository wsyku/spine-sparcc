# SpineSPARCC

**SpineSPARCC** is an inference-only research pipeline for sagittal
fat-suppressed spine MRI. It performs BME segmentation with a five-fold
nnU-Net ensemble, slice- and segment-level BME classification, segment-level
SPARCC score regression, and patient-level aggregation.

The repository includes one anonymized lumbar-segment example. Training data,
dataset splits, optimizer states, training logs, and historical experiments
are not distributed.

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

The included lumbar example can be run directly. Additional locally authorized
images may be added to `examples/input` using the naming convention above.
Run from the repository root:

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

## Model details

The segmentation stage uses a five-fold nnU-Net ensemble to generate a BME
mask and foreground probability map. The downstream models use the MRI
multiplied elementwise by that probability map.

### BME classifier

- Input: 16 central sagittal slices resized to 320 x 320.
- Backbone: randomly initialized 2D ResNet18.
- Slice branch: a shared linear head over 512-dimensional slice features.
- Segment branch: adjacent slice features are averaged, normalized with
  `LayerNorm(512)`, concatenated with a learned 16-dimensional C/T/L embedding,
  and passed through a shared `528 -> 128 -> 1` pair head.
- Pooling: softmax-weighted sum of valid pair logits with temperature 0.5.
- Checkpoint: fold 0 selected by minimum combined validation loss.

### SPARCC regressor

- Input and spatial preprocessing are identical to the classifier.
- Backbone: randomly initialized 2D ResNet34.
- A learned 512-dimensional C/T/L embedding is added to each slice feature,
  followed by `LayerNorm(512)` and dropout 0.1.
- Attention pooling: `512 -> 128 -> 1` with Tanh and softmax over valid slices.
- Regression head: `128 -> 256 -> 128 -> 64 -> 1` with GELU, LayerNorm, and
  dropout 0.3.
- Softplus produces a non-negative raw score; released inference clips it to
  the valid 0-108 range.
- Checkpoint: fold 0 selected by maximum validation ICC(2,1) among
  reference-positive segments.

Only segmentation uses five-fold ensembling. Classification and regression
each use one selected fold-0 checkpoint. The final segment score is the bounded
raw regression score when the segment classifier is positive and zero
otherwise; both raw and gated scores are exported.

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

## Intended use and limitations

SpineSPARCC is intended for reproduction of the accompanying research
pipeline, methodological research, and non-clinical benchmarking on
appropriately governed MRI data. It has not been validated as a clinical
diagnostic device and does not replace expert image interpretation.

Performance may change with scanner, acquisition protocol, field strength,
anatomy coverage, orientation, intensity distribution, and disease prevalence.
Outputs require independent clinical and statistical validation before use in
a new cohort.

The repository includes one anonymized lumbar-segment example with a generic
identifier and cleared descriptive NIfTI header fields. Users are responsible
for ensuring that additional inputs comply with applicable ethics, consent,
privacy, and data-governance requirements.

## Citation and license

See `CITATION.cff` for citation metadata and `LICENSE` for the code license.
Model weights are distributed under the terms supplied with the corresponding
GitHub Release. No clinical-use rights are granted.

### Third-party components

SpineSPARCC depends on and/or adapts components from the projects below. Their
respective licenses continue to apply.

- **nnU-Net v2** — [project](https://github.com/MIC-DKFZ/nnUNet), Apache
  License 2.0. See Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH.
  *nnU-Net: a self-configuring method for deep learning-based biomedical image
  segmentation.* Nature Methods. 2021;18:203–211.
- **dynamic-network-architectures** —
  [project](https://github.com/MIC-DKFZ/dynamic-network-architectures), Apache
  License 2.0.
- **PyTorch** and **torchvision** —
  [PyTorch](https://github.com/pytorch/pytorch) and
  [torchvision](https://github.com/pytorch/vision), under the BSD-style
  licenses supplied by their respective projects.
- **Other Python dependencies** — NumPy, pandas, NiBabel, and PyYAML remain
  subject to their own licenses. Released version pins are listed in
  `requirements.txt`.
