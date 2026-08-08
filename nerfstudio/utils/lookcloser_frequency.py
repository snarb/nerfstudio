"""Automatic frequency-map selection from progressive HDR reconstruction curves."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class FrequencyMapQuality:
    spearman: float
    detail_overlap: float
    spatial_coherence: float
    normalized_entropy: float
    effective_bins: float
    max_bin_fraction: float
    top2_bin_fraction: float
    unresolved_fraction: float
    nonempty_bins: int

    def objectives(self) -> np.ndarray:
        """All returned objectives are oriented so larger is better."""

        return np.asarray(
            [
                self.spearman,
                self.detail_overlap,
                self.spatial_coherence,
                self.normalized_entropy,
                np.log(max(self.effective_bins, 1.0)),
                -self.max_bin_fraction,
                -self.top2_bin_fraction,
                -self.unresolved_fraction,
            ],
            dtype=np.float64,
        )

    def as_dict(self) -> dict:
        return asdict(self)


def monotonic_recovery(scores: Tensor) -> Tensor:
    """Return a non-decreasing envelope along the first (frequency-level) axis."""

    if scores.ndim < 2:
        raise ValueError("scores must have shape (levels, ...)")
    return torch.cummax(scores.float(), dim=0).values


def first_crossing_levels(scores: Tensor, threshold: float) -> Tuple[Tensor, Tensor]:
    """Assign the first level crossing a data-derived absolute threshold."""

    envelope = monotonic_recovery(scores)
    crossed = envelope >= float(threshold)
    levels = crossed.float().argmax(dim=0).to(torch.int64)
    unresolved = ~crossed.any(dim=0)
    levels[unresolved] = envelope.shape[0] - 1
    return levels, unresolved


def relative_ensemble_levels(
    scores: Tensor,
    center: float,
    half_width: float = 0.15,
    flat_epsilon: float = 1e-4,
) -> Tuple[Tensor, Tensor]:
    """Aggregate three relative recovery crossings without an absolute SSIM threshold."""

    envelope = monotonic_recovery(scores)
    dynamic = envelope[-1] - envelope[0]
    normalized = (envelope - envelope[0]) / dynamic.clamp_min(float(flat_epsilon))
    thresholds = [max(0.05, center - half_width), center, min(0.95, center + half_width)]
    crossings = []
    unresolved_any = torch.zeros_like(dynamic, dtype=torch.bool)
    for threshold in thresholds:
        crossed = normalized >= float(threshold)
        level = crossed.float().argmax(dim=0).to(torch.int64)
        unresolved = ~crossed.any(dim=0)
        level[unresolved] = envelope.shape[0] - 1
        crossings.append(level)
        unresolved_any |= unresolved
    levels = torch.stack(crossings, dim=0).median(dim=0).values
    flat = dynamic < float(flat_epsilon)
    levels[flat] = 0
    unresolved_any[flat] = False
    return levels, unresolved_any


def knee_levels(scores: Tensor, flat_epsilon: float = 1e-4) -> Tuple[Tensor, Tensor]:
    """Select a threshold-free saturation knee using distance above the diagonal."""

    envelope = monotonic_recovery(scores)
    dynamic = envelope[-1] - envelope[0]
    normalized = (envelope - envelope[0]) / dynamic.clamp_min(float(flat_epsilon))
    x = torch.linspace(0.0, 1.0, envelope.shape[0], device=envelope.device, dtype=envelope.dtype)
    x = x.view(-1, *([1] * (envelope.ndim - 1)))
    levels = torch.argmax(normalized - x, dim=0).to(torch.int64)
    flat = dynamic < float(flat_epsilon)
    levels[flat] = 0
    return levels, torch.zeros_like(flat, dtype=torch.bool)


def levels_to_resolutions(levels: Tensor, min_res: float, max_res: float, n_levels: int) -> Tensor:
    if n_levels < 2 or min_res <= 0 or max_res <= min_res:
        raise ValueError("Expected n_levels >= 2 and 0 < min_res < max_res")
    scale = float(np.exp((np.log(max_res) - np.log(min_res)) / (n_levels - 1)))
    return float(min_res) * torch.pow(torch.as_tensor(scale, dtype=torch.float32), levels.float())


def guided_median_levels(levels: Tensor, structural_proxy: Tensor, detail_fraction: float = 0.2) -> Tensor:
    """Suppress isolated bin noise while preserving the strongest structural patches."""

    if levels.ndim != 2 or levels.shape != structural_proxy.shape:
        raise ValueError("guided median expects matching 2D levels and structural proxy")
    padded = torch.nn.functional.pad(levels[None, None].float(), (1, 1, 1, 1), mode="replicate")
    median = torch.nn.functional.unfold(padded, kernel_size=3).median(dim=1).values.reshape_as(levels)
    cutoff = torch.quantile(structural_proxy.float(), 1.0 - float(detail_fraction))
    preserve = structural_proxy >= cutoff
    return torch.where(preserve, levels, median.to(levels.dtype))


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    return ranks


def map_quality(
    levels: Tensor,
    structural_proxy: Tensor,
    *,
    n_levels: int,
    unresolved: Tensor | None = None,
    detail_fraction: float = 0.2,
) -> FrequencyMapQuality:
    """Score how well a map represents structure while avoiding distribution collapse."""

    if levels.shape != structural_proxy.shape:
        raise ValueError("levels and structural_proxy must have the same shape")
    level_values = levels.detach().cpu().numpy().reshape(-1).astype(np.float64)
    proxy_values = structural_proxy.detach().cpu().numpy().reshape(-1).astype(np.float64)
    if level_values.size == 0 or not np.isfinite(proxy_values).all():
        raise ValueError("map quality requires non-empty finite inputs")
    level_rank = _rank(level_values)
    proxy_rank = _rank(proxy_values)
    if np.std(level_rank) == 0 or np.std(proxy_rank) == 0:
        spearman = 0.0
    else:
        spearman = float(np.corrcoef(level_rank, proxy_rank)[0, 1])
    top_count = max(1, int(np.ceil(detail_fraction * level_values.size)))
    detail_indices = set(np.argpartition(proxy_values, -top_count)[-top_count:].tolist())
    map_indices = set(np.argpartition(level_values, -top_count)[-top_count:].tolist())
    overlap = float(len(detail_indices & map_indices) / top_count)
    same_horizontal = (
        (levels[:, 1:] == levels[:, :-1]).float().mean() if levels.shape[1] > 1 else levels.new_tensor(1.0)
    )
    same_vertical = (
        (levels[1:, :] == levels[:-1, :]).float().mean() if levels.shape[0] > 1 else levels.new_tensor(1.0)
    )
    spatial_coherence = float(0.5 * (same_horizontal + same_vertical))

    counts = np.bincount(level_values.astype(np.int64), minlength=n_levels).astype(np.float64)
    probabilities = counts / max(counts.sum(), 1.0)
    nonzero = probabilities[probabilities > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    normalized_entropy = entropy / np.log(max(n_levels, 2))
    sorted_probabilities = np.sort(probabilities)[::-1]
    unresolved_fraction = float(unresolved.float().mean().item()) if unresolved is not None else 0.0
    return FrequencyMapQuality(
        spearman=spearman,
        detail_overlap=overlap,
        spatial_coherence=spatial_coherence,
        normalized_entropy=normalized_entropy,
        effective_bins=float(np.exp(entropy)),
        max_bin_fraction=float(sorted_probabilities[0]),
        top2_bin_fraction=float(sorted_probabilities[:2].sum()),
        unresolved_fraction=unresolved_fraction,
        nonempty_bins=int(np.count_nonzero(counts)),
    )


def balanced_topsis(objectives: np.ndarray) -> np.ndarray:
    """Robust-normalized, equal-weight TOPSIS closeness for candidate rows."""

    matrix = np.asarray(objectives, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("objectives must be a non-empty 2D array")
    median = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - median), axis=0)
    robust = (matrix - median) / np.where(mad > 1e-12, 1.4826 * mad, 1.0)
    minimum = robust.min(axis=0)
    span = robust.max(axis=0) - minimum
    normalized = (robust - minimum) / np.where(span > 1e-12, span, 1.0)
    ideal = normalized.max(axis=0)
    anti_ideal = normalized.min(axis=0)
    distance_ideal = np.linalg.norm(normalized - ideal, axis=1)
    distance_anti = np.linalg.norm(normalized - anti_ideal, axis=1)
    return distance_anti / np.maximum(distance_ideal + distance_anti, 1e-12)


def bootstrap_select(
    qualities: Mapping[str, Sequence[FrequencyMapQuality]],
    *,
    resamples: int = 200,
    seed: int = 0,
) -> Dict[str, object]:
    """Select a scene-level candidate by bootstrap image resampling and TOPSIS."""

    names = sorted(qualities)
    if not names:
        raise ValueError("No candidates to select")
    image_count = len(qualities[names[0]])
    if image_count == 0 or any(len(qualities[name]) != image_count for name in names):
        raise ValueError("Each candidate must have one quality row per image")
    objective_cube = np.stack(
        [np.stack([quality.objectives() for quality in qualities[name]], axis=0) for name in names], axis=0
    )
    rng = np.random.default_rng(seed)
    wins = np.zeros(len(names), dtype=np.int64)
    closeness_rows = []
    for _ in range(max(1, resamples)):
        indices = rng.integers(0, image_count, size=image_count)
        mean_objectives = objective_cube[:, indices].mean(axis=1)
        closeness = balanced_topsis(mean_objectives)
        closeness_rows.append(closeness)
        wins[int(np.argmax(closeness))] += 1
    closeness_array = np.stack(closeness_rows, axis=0)
    median_closeness = np.median(closeness_array, axis=0)
    winner_index = max(range(len(names)), key=lambda idx: (wins[idx], median_closeness[idx], names[idx]))
    return {
        "winner": names[winner_index],
        "wins": {name: int(wins[idx]) for idx, name in enumerate(names)},
        "median_closeness": {name: float(median_closeness[idx]) for idx, name in enumerate(names)},
        "resamples": int(max(1, resamples)),
        "seed": int(seed),
    }
