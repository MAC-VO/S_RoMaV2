import torch

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# ONNX-export precision policy.
#
# By default the internal bf16 autocast regions are DISABLED during tracing/export
# (`torch.jit.is_tracing()` guards), producing a clean all-fp32 ONNX graph used by the
# permissive `BuilderFlag.BF16` TensorRT build (TRT then picks bf16 per layer).
#
# When `keep_autocast_in_trace()` is True, the autocast regions stay ENABLED during
# export, so the exported graph is *mixed precision*: matmul/conv run bf16 (autocast only
# casts those), while elementwise math (softplus, the Cholesky squaring, accumulation)
# stays fp32. Building a STRONGLY-TYPED engine from that graph reproduces eager RoMa's
# exact precision — bf16 backbone + fp32 confidence path — which per-layer precision pins
# cannot achieve because TensorRT fuses the confidence tail into bf16 Myelin blocks that
# silently ignore the pins.
_KEEP_AUTOCAST_IN_TRACE = False


def keep_autocast_in_trace() -> bool:
    return _KEEP_AUTOCAST_IN_TRACE


def set_keep_autocast_in_trace(value: bool) -> None:
    global _KEEP_AUTOCAST_IN_TRACE
    _KEEP_AUTOCAST_IN_TRACE = value
