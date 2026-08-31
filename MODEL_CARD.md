# SpineSPARCC model card

## Summary

SpineSPARCC is an inference-only research pipeline for bone marrow edema (BME)
localization, slice/segment classification, and SPARCC score estimation from
sagittal fat-suppressed spine MRI.

## Pipeline

```text
Sagittal MRI
└── Dataset351 nnU-Net five-fold ensemble
    ├── BME mask
    └── BME foreground probability
        ├── slice and segment classifier
        └── SPARCC regressor
```

## Classifier

- Input: MRI multiplied elementwise by the BME probability map.
- Shape: 16 central sagittal slices resized to 320 x 320.
- Backbone: randomly initialized 2D ResNet18.
- Slice output: shared linear head over 512-dimensional slice features.
- Segment branch: adjacent slice features are averaged, normalized with
  `LayerNorm(512)`, concatenated with a learned 16-dimensional C/T/L
  embedding, and passed through a shared `528 -> 128 -> 1` pair head.
- Pooling: softmax-weighted sum of pair logits with temperature 0.5.
- Checkpoint: selected fold 0 by minimum combined validation loss.
- Segment threshold: 0.6978082060813904, selected on validation data.

## Regressor

- Input and spatial preprocessing are identical to the classifier.
- Backbone: randomly initialized 2D ResNet34.
- A learned 512-dimensional C/T/L embedding is added to each slice feature,
  followed by `LayerNorm(512)` and dropout 0.1.
- Attention pooling: `512 -> 128 -> 1` with Tanh and softmax over valid slices.
- Regression head: `128 -> 256 -> 128 -> 64 -> 1` with GELU, LayerNorm, and
  dropout 0.3.
- Softplus ensures a non-negative raw prediction; released inference clips the
  score to the valid 0-108 range.
- Checkpoint: selected fold 0 by maximum validation ICC(2,1) among
  reference-positive segments.

## Ensembling and final score

Only the nnU-Net segmentation model uses five-fold ensembling. The classifier
and regressor use one selected fold-0 checkpoint each; there is no probability,
logit, or score averaging across classifier/regressor folds.

The final segment score is the bounded raw regression score when the segment
classifier is positive and zero otherwise. Both raw and gated scores are
exported.

## Intended use

- Reproduction of the accompanying research pipeline.
- Methodological research on automated spinal inflammation quantification.
- Non-clinical benchmarking on appropriately governed MRI data.

## Limitations

- Not validated as a clinical diagnostic device.
- Performance may change with scanner, acquisition protocol, field strength,
  anatomy coverage, orientation, intensity distribution, and disease
  prevalence.
- The released classifier/regressor are single-fold checkpoints.
- Outputs require independent clinical and statistical validation before use
  in a new cohort.
- The model does not replace expert image interpretation.

## Data and privacy

The repository includes one anonymized lumbar-segment example with a generic
identifier and cleared descriptive NIfTI header fields. Users must ensure that
any additional inputs comply with their ethics, consent, privacy, and
data-governance requirements.

## License

Source code is licensed under Apache-2.0. Model weights may be used only under
the release terms supplied with the corresponding GitHub Release. No
clinical-use rights are granted.
