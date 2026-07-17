"""Independent deterministic RNG streams for static LookCloser training."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Union

import torch


_STREAM_IDS = {"pixel": 1, "occupancy": 2, "frequency_grid": 3}


def stream_seed(base_seed: int, stream: str, step: int) -> int:
    """Return a stable seed without depending on Python's randomized hash()."""
    if stream not in _STREAM_IDS:
        raise ValueError(f"Unknown LookCloser RNG stream: {stream}")
    if step < 0:
        raise ValueError("LookCloser RNG stream step must be non-negative.")
    return (int(base_seed) + _STREAM_IDS[stream] * 104_729 + int(step) * 1_000_003) % (2**31)


@contextmanager
def fork_seeded_rng(
    base_seed: int,
    stream: str,
    step: int,
    device: Union[str, torch.device],
) -> Iterator[None]:
    """Fork CPU/CUDA RNG state and seed one repeatable module-local step stream."""
    resolved = torch.device(device)
    devices = []
    if resolved.type == "cuda":
        devices = [resolved.index if resolved.index is not None else torch.cuda.current_device()]
    seed = stream_seed(base_seed, stream, step)
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.default_generators[devices[0]].manual_seed(seed)
        yield
