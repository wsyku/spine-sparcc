# Third-party notices

SpineSPARCC depends on and/or adapts components from the following projects.
Their licenses apply to their respective components.

## nnU-Net v2

- Project: https://github.com/MIC-DKFZ/nnUNet
- License: Apache License 2.0
- Citation: Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH.
  *nnU-Net: a self-configuring method for deep learning-based biomedical image
  segmentation.* Nature Methods. 2021;18:203-211.
- `src/nnunet_network_factory.py` is adapted from nnU-Net v2.8.0 network
  construction utilities and has been modified for the packaged plans file.

## dynamic-network-architectures

- Project: https://github.com/MIC-DKFZ/dynamic-network-architectures
- License: Apache License 2.0

## PyTorch and torchvision

- Projects: https://github.com/pytorch/pytorch and
  https://github.com/pytorch/vision
- Licenses: BSD-style licenses supplied by the respective projects.

## Other Python dependencies

NumPy, pandas, NiBabel, and PyYAML are installed as dependencies and remain
subject to their own licenses. See `requirements.txt` for the released version
pins.
