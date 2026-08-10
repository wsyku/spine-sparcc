# Model files

The inference scripts expect this directory layout:

```text
models/
├── checksums.sha256
├── classification/
│   └── model.pth
├── regression/
│   └── model.pth
└── segmentation/
    ├── dataset.json
    ├── plans.json
    └── fold_0.pth ... fold_4.pth
```

Model weights are not tracked in the normal Git history. Download
`spine-sparcc-models-v0.1.0.zip` from GitHub Release `v0.1.0` and extract its
contents directly into this `models` directory.

Verify the files before inference:

```powershell
python tools/verify_release.py
```

The verifier checks all seven weight files against `checksums.sha256` and also
checks the required configuration, metadata, and example inputs.
