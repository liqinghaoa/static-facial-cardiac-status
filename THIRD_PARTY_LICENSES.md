# Third-party source notices

This package retains upstream source code and public demonstration inputs for
the R3DPR preprocessing path. It does not redistribute external model weights,
BFM files, or study-trained checkpoints. Each retained component remains under
its original terms; this file records the locations that need attribution.

| Component | Retained location | Notice or license boundary |
| --- | --- | --- |
| PortraitRelighting / R3DPR | `vendor/PortraitRelighting/` | Apache License 2.0; retain the accompanying `LICENSE` and upstream attribution. |
| nvdiffrast | `vendor/nvdiffrast/` | NVIDIA Source Code License (1-Way Commercial); retain `LICENSE.txt`. |
| CropPose / Deep3DFaceRecon support | `vendor/PortraitRelighting/third_party/CropPose/` | Follow the original upstream terms and attribution. BFM and model files are not included. |
| DPR | `vendor/PortraitRelighting/third_party/DPR/` | Retain the upstream README and its research-purpose notice. |
| NeRFFaceLighting / EG3D support | `vendor/PortraitRelighting/third_party/NeRFFaceLighting/` | Follow the nested upstream terms and attribution retained with the source. |
| Demonstration and illumination inputs | `vendor/PortraitRelighting/examples/` and `vendor/PortraitRelighting/third_party/DPR/data/example_light/` | Upstream examples only; they are not participant data or study results. The original upstream terms apply. |

This notice does not replace any upstream license. Users who redistribute or
modify the retained third-party source must preserve the corresponding files
and comply with the original terms.
