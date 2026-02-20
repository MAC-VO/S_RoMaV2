"""
RoMaV2 DINO local descriptor extraction utilities for VPR.
"""

from __future__ import annotations

import typing as T

import torch
import torch.nn.functional as F


T_RoMaFacet = T.Literal["token", "key", "query", "value"]


_VALID_FACETS: set[str] = {"token", "key", "query", "value"}
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _preprocess_image(
    image_l: torch.Tensor,
    *,
    max_img_size: int,
    patch_size: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    assert image_l.dim() == 3, f"Expected image_l shape [3, H, W], got {tuple(image_l.shape)}."
    assert int(image_l.size(0)) == 3, f"Expected image_l channel size = 3, got {int(image_l.size(0))}."
    assert max_img_size > 0, "max_img_size must be positive."
    assert patch_size > 0, "patch_size must be positive."

    image_l = image_l.detach().cpu().to(dtype=torch.float32)

    if max(image_l.shape[-2:]) > max_img_size:
        _, img_h, img_w = image_l.shape
        if img_h >= img_w:
            out_h = max_img_size
            out_w = int(round(img_w * (max_img_size / img_h)))
        else:
            out_w = max_img_size
            out_h = int(round(img_h * (max_img_size / img_w)))
        image_l = F.interpolate(
            image_l.unsqueeze(0),
            size=(out_h, out_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    _, img_h, img_w = image_l.shape
    crop_h = (img_h // patch_size) * patch_size
    crop_w = (img_w // patch_size) * patch_size
    assert (crop_h > 0) and (crop_w > 0), (
        "RoMaV2 local descriptor extraction requires image spatial size "
        f">= {patch_size}x{patch_size} after resizing."
    )

    start_h = (img_h - crop_h) // 2
    start_w = (img_w - crop_w) // 2
    image_l = image_l[:, start_h:start_h + crop_h, start_w:start_w + crop_w]

    image_l = (image_l - _IMAGENET_MEAN) / _IMAGENET_STD
    return image_l, (crop_h // patch_size, crop_w // patch_size)


def _extract_hook_tensor(payload: torch.Tensor | tuple[torch.Tensor, ...] | None) -> torch.Tensor:
    if payload is None:
        raise RuntimeError("RoMaV2 descriptor hook did not capture any output tensor.")
    if isinstance(payload, tuple):
        assert len(payload) > 0, "RoMaV2 descriptor hook returned an empty tuple."
        payload = payload[0]
    if not isinstance(payload, torch.Tensor):
        raise RuntimeError(f"RoMaV2 descriptor hook returned type={type(payload)} instead of torch.Tensor.")
    return payload


def extract_roma_local_descriptors(
    *,
    descriptor_model: torch.nn.Module,
    image_l: torch.Tensor,
    layer: int,
    facet: T_RoMaFacet,
    max_img_size: int,
    device: str | torch.device,
    patch_size: int = 16,
) -> torch.Tensor:
    """
    Extract local descriptors from RoMaV2 DINO backbone with configurable layer/facet.

    Returns:
        Tensor of shape [N, D], L2-normalized in float32 on CPU.
    """
    if facet not in _VALID_FACETS:
        raise ValueError(f"Unsupported RoMa descriptor facet='{facet}'. Expected one of {_VALID_FACETS}.")
    if layer < 0:
        raise ValueError(f"desc_layer must be >= 0, got {layer}.")

    if not hasattr(descriptor_model, "blocks"):
        raise ValueError("RoMa descriptor model does not expose 'blocks'; cannot register layer hooks.")
    if not hasattr(descriptor_model, "get_intermediate_layers"):
        raise ValueError(
            "RoMa descriptor model does not expose 'get_intermediate_layers'; "
            "cannot run descriptor extraction."
        )

    blocks = getattr(descriptor_model, "blocks")
    if layer >= len(blocks):
        raise ValueError(
            f"desc_layer={layer} is out of range for descriptor model with {len(blocks)} blocks."
        )

    image_l, (patch_h, patch_w) = _preprocess_image(
        image_l,
        max_img_size=max_img_size,
        patch_size=patch_size,
    )
    expected_patch_tokens = patch_h * patch_w

    first_param = next(descriptor_model.parameters())
    model_device = first_param.device
    model_dtype = first_param.dtype
    infer_device = torch.device(device)

    if model_device.type != infer_device.type:
        raise RuntimeError(
            "RoMa descriptor model device type does not match requested extraction device type: "
            f"model_device={model_device}, requested_device={infer_device}."
        )

    if model_device.type == "cuda":
        # Treat 'cuda' and 'cuda:0' as equivalent when model is on cuda:0.
        req_idx = infer_device.index
        model_idx = model_device.index
        if req_idx is None:
            infer_device = model_device
        elif (model_idx is not None) and (req_idx != model_idx):
            raise RuntimeError(
                "RoMa descriptor model CUDA device index does not match requested extraction index: "
                f"model_device={model_device}, requested_device={infer_device}."
            )
    else:
        infer_device = model_device

    hook_output: dict[str, torch.Tensor | tuple[torch.Tensor, ...] | None] = {"value": None}

    def _capture_hook(_module, _inputs, output) -> None:
        hook_output["value"] = output

    target_block = blocks[layer]
    if facet == "token":
        handle = target_block.register_forward_hook(_capture_hook)
    else:
        handle = target_block.attn.qkv.register_forward_hook(_capture_hook)

    try:
        with torch.inference_mode():
            model_input = image_l.unsqueeze(0).to(device=infer_device, dtype=model_dtype)
            descriptor_model.get_intermediate_layers(
                model_input,
                n=[layer],
            )
    finally:
        handle.remove()

    hooked = _extract_hook_tensor(hook_output["value"])
    if hooked.dim() != 3:
        raise RuntimeError(
            "RoMa descriptor hook tensor must be rank-3 [B, N, C], "
            f"got shape={tuple(hooked.shape)}."
        )

    batch_size, token_count, channel_dim = map(int, hooked.shape)
    if batch_size != 1:
        raise RuntimeError(
            "RoMa descriptor extraction expects unbatched input (B=1), "
            f"but hook tensor has B={batch_size}."
        )

    if facet in {"query", "key", "value"}:
        if channel_dim % 3 != 0:
            raise RuntimeError(
                "Expected qkv hook channel dimension divisible by 3, "
                f"got C={channel_dim}."
            )
        dim_per_facet = channel_dim // 3
        match facet:
            case "query":
                hooked = hooked[:, :, :dim_per_facet]
            case "key":
                hooked = hooked[:, :, dim_per_facet:2 * dim_per_facet]
            case "value":
                hooked = hooked[:, :, 2 * dim_per_facet:]
            case _:
                raise ValueError(f"Unsupported facet '{facet}'.")

    local_descriptors = hooked[0]

    if token_count > expected_patch_tokens:
        # DINOv3 can prepend extra tokens (e.g. CLS + register tokens).
        # Keep only patch tokens.
        num_prefix_tokens = token_count - expected_patch_tokens
        local_descriptors = local_descriptors[num_prefix_tokens:]
    elif token_count < expected_patch_tokens:
        raise RuntimeError(
            "Unexpected token count from RoMa descriptor hook. "
            f"got={token_count}, expected_at_least={expected_patch_tokens}."
        )

    if int(local_descriptors.size(0)) != expected_patch_tokens:
        raise RuntimeError(
            "RoMa descriptor extraction produced invalid patch descriptor count: "
            f"got={int(local_descriptors.size(0))}, expected={expected_patch_tokens}."
        )

    local_descriptors = local_descriptors.detach().to(dtype=torch.float32)
    local_descriptors = F.normalize(local_descriptors, dim=1)
    return local_descriptors.cpu()
