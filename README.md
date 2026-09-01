# SpineSPARCC

SpineSPARCC estimates spinal bone marrow edema (BME) and SPARCC scores from
sagittal fat-suppressed MRI. The inference pipeline contains three separately
trained models:

1. a five-fold nnU-Net ensemble for BME segmentation;
2. a ResNet18 classifier for slice- and segment-level BME prediction;
3. a ResNet34 regressor for segment-level SPARCC scoring.

The nnU-Net foreground probability map is multiplied by the MR image before it
is passed to the classifier and regressor. If the classifier marks a segment as
BME-negative, its final SPARCC score is set to zero. Raw regression scores are
also saved.

This repository contains inference code and one anonymized lumbar example. It
does not include training data, data splits, optimizer states, training logs,
or experimental code. The software is intended for research use and is not a
medical device.

## Installation

The reference environment used for release testing was Python 3.12.13,
PyTorch 2.12.1, torchvision 0.27.1, CUDA 13.2, nnU-Net 2.8.0, Windows 11, and
an NVIDIA RTX 5090 GPU.

```powershell
git clone https://github.com/wsyku/spine-sparcc.git
cd spine-sparcc
python -m pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA version if it differs from
the reference environment. CPU inference is supported but is considerably
slower.

## Model weights

Download `spine-sparcc-models-v0.1.0.zip` from the GitHub Release and extract
it into `models/`. The expected directory structure is described in
[`models/README.md`](models/README.md).

The files can be checked after extraction with:

```powershell
python tools/verify_release.py
```

## Input

Input images must be NIfTI files. The filename identifies the patient and
spinal region:

```text
<patient_id>_C_0000.nii.gz
<patient_id>_T_0000.nii.gz
<patient_id>_L_0000.nii.gz
```

`C`, `T`, and `L` refer to the cervical, thoracic, and lumbar spine. It is not
necessary to provide all three regions for every patient.

## Inference

Run the included example from the repository root:

```powershell
python src/run_inference.py --input examples/input --output examples/output
```

The output directory contains:

```text
examples/output/
├── segmentation/
├── slice_classification.csv
├── segment_results.csv
└── patient_summary.csv
```

- `segmentation/` contains the BME masks.
- `slice_classification.csv` contains slice probabilities and labels.
- `segment_results.csv` contains segment probabilities, raw regression scores,
  and classification-gated SPARCC scores.
- `patient_summary.csv` contains the summed SPARCC score for each patient.

The segment threshold is `0.6978082060813904`. It was selected on the
validation set and is not recalculated during inference. Regression scores are
limited to the valid SPARCC range of 0–108.

## Model and preprocessing details

Volumes are centrally cropped to 16 slices or padded at the end when fewer
slices are available. Padded slices and invalid adjacent-slice pairs are
excluded. MR images and probability maps are resized to 320 × 320 using
bilinear interpolation. MR intensities greater than zero are normalized by
their foreground mean and standard deviation.

The classifier uses a 2D ResNet18. Its slice head operates on individual
512-dimensional features. For segment prediction, adjacent slice features are
averaged and normalized, then concatenated with a learned 16-dimensional
cervical/thoracic/lumbar embedding. A shared pair head produces logits that are
combined by softmax pooling with temperature 0.5. The released checkpoint was
selected by the lowest combined validation loss.

The regressor uses a 2D ResNet34. A learned 512-dimensional region embedding is
added to each slice feature before LayerNorm and attention pooling. The pooled
feature is mapped to a non-negative score and clipped to 0–108. The released
checkpoint was selected by the highest validation ICC(2,1) among
reference-positive segments.

Only the segmentation stage uses five-fold ensembling. The released classifier
and regressor each use one fold-0 checkpoint.

Inference uses seed 42 with deterministic cuDNN settings. Small numerical
differences may still occur between PyTorch, CUDA, and GPU versions.

## Limitations and data use

Performance can vary with scanner, acquisition protocol, field strength,
anatomical coverage, orientation, and disease prevalence. Results should be
validated independently before the model is applied to a new cohort.

The example image has a generic filename and cleared descriptive NIfTI header
fields. It is provided only to demonstrate inference and is not covered by the
Apache-2.0 code license. See
[`examples/DATA_NOTICE.md`](examples/DATA_NOTICE.md). Users are responsible for
the governance and authorization of any additional images.

## Citation and licenses

Citation information is provided in [`CITATION.cff`](CITATION.cff). Source code
is released under the [Apache License 2.0](LICENSE). Terms for model weights are
provided with the GitHub Release.

SpineSPARCC uses the following third-party projects under their respective
licenses:

- [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet), Apache License 2.0. See:
  Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. *nnU-Net: a
  self-configuring method for deep learning-based biomedical image
  segmentation.* Nature Methods. 2021;18:203–211.
- [dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures),
  Apache License 2.0.
- [PyTorch](https://github.com/pytorch/pytorch) and
  [torchvision](https://github.com/pytorch/vision), under their distributed
  BSD-style licenses.

Other dependencies are listed in [`requirements.txt`](requirements.txt) and
remain subject to their own licenses.
