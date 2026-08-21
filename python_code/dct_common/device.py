"""GPU-for-training / CPU-for-inference device helpers, shared by both scripts.

Training uses whatever `get_device()` returns (GPU if available). Final
inference latency is always benchmarked on CPU specifically, since that's
this project's actual deployment target (an ESP32-S3, not this machine's
GPU) -- `benchmark_cpu_inference` explicitly moves the model to CPU before
timing, regardless of where it was trained.
"""

import time

import torch
from torch import nn


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA GPU"
    print(f"Training device: {device} | {name}")
    return device


def benchmark_cpu_inference(model: nn.Module, sample_inputs: tuple, n_warmup: int = 20, n_trials: int = 200) -> dict:
    """Single-image forward-pass latency on CPU. `sample_inputs` is a tuple
    of already-batch-1 tensors (e.g. `(x,)` for the MLP, `(x, x_chroma)`
    for the CNN); returns {"ms_per_inference": ..., "fps": ...}. Mutates
    `model` in place onto CPU as a side effect."""
    model = model.to("cpu").eval()
    sample_inputs = tuple(t.to("cpu") for t in sample_inputs)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(*sample_inputs)
        start = time.perf_counter()
        for _ in range(n_trials):
            model(*sample_inputs)
        elapsed = time.perf_counter() - start
    ms_per_call = elapsed / n_trials * 1000.0
    return {"ms_per_inference": ms_per_call, "fps": 1000.0 / ms_per_call}


def benchmark_callable(forward_fn, sample_inputs: tuple, n_warmup: int = 20, n_trials: int = 200) -> dict:
    """Same timing methodology as `benchmark_cpu_inference`, for a plain
    Python/NumPy callable instead of a torch model -- e.g. the bit-exact
    int8 reference forward pass, which is what actually runs on real
    firmware (unlike the float model's PyTorch latency, which only tells
    you about this machine's PyTorch overhead)."""
    for _ in range(n_warmup):
        forward_fn(*sample_inputs)
    start = time.perf_counter()
    for _ in range(n_trials):
        forward_fn(*sample_inputs)
    elapsed = time.perf_counter() - start
    ms_per_call = elapsed / n_trials * 1000.0
    return {"ms_per_inference": ms_per_call, "fps": 1000.0 / ms_per_call}
