import torch


# Module-level constants so the normalization tensors are built once at import time
# rather than via `torch.tensor([...])` inside the forward — the latter emits a
# "torch.tensor results are registered as constants" TracerWarning during ONNX export.
# Inside forward we only move them onto the input's device (a plain cast, no warning).
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])
_INCEPTION_MEAN = torch.tensor([0.5, 0.5, 0.5])
_INCEPTION_STD = torch.tensor([0.5, 0.5, 0.5])


def imagenet(img: torch.Tensor) -> torch.Tensor:
    imagenet_mean = _IMAGENET_MEAN.to(img.device)
    imagenet_std = _IMAGENET_STD.to(img.device)
    return (img - imagenet_mean[None, :, None, None]) / imagenet_std[
        None, :, None, None
    ]


def inception(img: torch.Tensor) -> torch.Tensor:
    inception_mean = _INCEPTION_MEAN.to(img.device)
    inception_std = _INCEPTION_STD.to(img.device)
    return (img - inception_mean[None, :, None, None]) / inception_std[
        None, :, None, None
    ]
