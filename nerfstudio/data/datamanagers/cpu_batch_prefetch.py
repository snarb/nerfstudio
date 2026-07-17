"""Private-generator, one-batch CPU pixel-sampling prefetch."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class _PrefetchTarget:
    step: int
    sample_count: int
    signature: Tuple[Any, ...]
    pre_rng_state: torch.Tensor
    explicit_seed: Optional[int] = None


@dataclass(frozen=True)
class _PrefetchedBatch:
    target: _PrefetchTarget
    batch: Dict[str, Any]
    post_rng_state: torch.Tensor


class DeterministicCPUBatchPrefetcher:
    """One worker and queue depth one, with historical and step-seeded modes.

    Historical calls clone submit-time global CPU RNG into the worker and only
    commit the resulting post-state when all transaction invariants still
    match. Seeded calls instead derive a private generator from an explicit
    per-step seed and never read or mutate live global RNG.
    """

    def __init__(
        self,
        *,
        sample_batch: Callable[[torch.Generator, int], Dict[str, Any]],
        fallback_sample_batch: Callable[[], Dict[str, Any]],
        get_sample_count: Callable[[], int],
        commit_sample_count: Callable[[int], None],
        get_signature: Callable[[], Tuple[Any, ...]],
        supported_signature: Tuple[Any, ...],
    ) -> None:
        self._sample_batch = sample_batch
        self._fallback_sample_batch = fallback_sample_batch
        self._get_sample_count = get_sample_count
        self._commit_sample_count = commit_sample_count
        self._get_signature = get_signature
        self._supported_signature = supported_signature
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lookcloser-fas-prefetch")
        self._future: Optional[Future[_PrefetchedBatch]] = None
        self._closed = False
        self.discard_count = 0

    @property
    def has_pending_batch(self) -> bool:
        return self._future is not None

    @staticmethod
    def _run_transaction(
        target: _PrefetchTarget,
        sample_batch: Callable[[torch.Generator, int], Dict[str, Any]],
    ) -> _PrefetchedBatch:
        generator = torch.Generator(device="cpu")
        generator.set_state(target.pre_rng_state)
        batch = sample_batch(generator, target.sample_count)
        return _PrefetchedBatch(target=target, batch=batch, post_rng_state=generator.get_state())

    @staticmethod
    def _rng_state_for_seed(seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        return generator.get_state()

    def _submit(self, step: int, explicit_seed: Optional[int] = None) -> bool:
        if self._closed:
            raise RuntimeError("CPU batch prefetcher is closed")
        if self._future is not None:
            raise RuntimeError("CPU batch prefetch queue depth exceeded one")
        signature = self._get_signature()
        if signature != self._supported_signature:
            return False
        target = _PrefetchTarget(
            step=int(step),
            sample_count=int(self._get_sample_count()),
            signature=signature,
            pre_rng_state=(
                torch.get_rng_state().clone()
                if explicit_seed is None
                else self._rng_state_for_seed(explicit_seed)
            ),
            explicit_seed=None if explicit_seed is None else int(explicit_seed),
        )
        self._future = self._executor.submit(self._run_transaction, target, self._sample_batch)
        return True

    def _target_is_current(
        self,
        target: _PrefetchTarget,
        step: int,
        explicit_seed: Optional[int] = None,
    ) -> bool:
        metadata_matches = (
            target.step == int(step)
            and target.sample_count == int(self._get_sample_count())
            and target.signature == self._get_signature()
        )
        if not metadata_matches:
            return False
        if explicit_seed is not None:
            return target.explicit_seed == int(explicit_seed)
        return target.explicit_seed is None and torch.equal(target.pre_rng_state, torch.get_rng_state())

    def next_batch(self, step: int) -> Dict[str, Any]:
        """Return the target-step batch and stage exactly one successor."""

        if self._future is None:
            if not self._submit(step):
                return self._fallback_sample_batch()
        future = self._future
        self._future = None
        assert future is not None
        transaction = future.result()
        if self._target_is_current(transaction.target, step):
            torch.set_rng_state(transaction.post_rng_state)
            self._commit_sample_count(transaction.target.sample_count + 1)
            batch = transaction.batch
        else:
            self.discard_count += 1
            batch = self._fallback_sample_batch()
        self._submit(int(step) + 1)
        return batch

    def next_batch_seeded(self, step: int, explicit_seed: int, next_explicit_seed: int) -> Dict[str, Any]:
        """Return one step-addressed private-generator batch without touching global RNG."""

        if self._future is None:
            if not self._submit(step, explicit_seed):
                raise RuntimeError("Seeded CPU FAS prefetch no longer matches its supported sampling signature")
        future = self._future
        self._future = None
        assert future is not None
        transaction = future.result()
        if self._target_is_current(transaction.target, step, explicit_seed):
            self._commit_sample_count(transaction.target.sample_count + 1)
            batch = transaction.batch
        else:
            self.discard_count += 1
            signature = self._get_signature()
            if signature != self._supported_signature:
                raise RuntimeError("Seeded CPU FAS prefetch no longer matches its supported sampling signature")
            target = _PrefetchTarget(
                step=int(step),
                sample_count=int(self._get_sample_count()),
                signature=signature,
                pre_rng_state=self._rng_state_for_seed(explicit_seed),
                explicit_seed=int(explicit_seed),
            )
            transaction = self._run_transaction(target, self._sample_batch)
            self._commit_sample_count(target.sample_count + 1)
            batch = transaction.batch
        if not self._submit(int(step) + 1, next_explicit_seed):
            raise RuntimeError("Seeded CPU FAS prefetch no longer matches its supported sampling signature")
        return batch

    def discard_pending(self) -> None:
        """Join and discard derived work; live RNG/count were never changed."""

        future = self._future
        self._future = None
        if future is None:
            return
        future.result()
        self.discard_count += 1

    def close(self) -> None:
        if self._closed:
            return
        self.discard_pending()
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
