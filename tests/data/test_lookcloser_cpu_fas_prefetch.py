import threading

import numpy as np
import pytest
import torch

from nerfstudio.data.datamanagers.cpu_batch_prefetch import DeterministicCPUBatchPrefetcher
from nerfstudio.lookcloser_pixel_sampler import LookCloserPixelSampler, LookCloserPixelSamplerConfig
from nerfstudio.utils.lookcloser_rng import fork_seeded_rng, stream_seed


def _sampler_and_batch(*, fas_strength: float = 0.75):
    config = LookCloserPixelSamplerConfig(
        num_rays_per_batch=64,
        num_levels=3,
        patch_size=4,
        stride=4,
        fas_strength=fas_strength,
        fas_ramp_steps=5,
    )
    sampler = LookCloserPixelSampler(config)
    sampler.is_initialized = True
    sampler._prefetch_data_version = 1
    sampler.probs = np.array([0.2, 0.3, 0.5])
    sampler.buckets = {
        0: torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int32),
        1: torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.int32),
        2: torch.tensor([[0, 1, 1], [1, 0, 0]], dtype=torch.int32),
    }
    sampler.image_shapes = {0: (8, 8), 1: (8, 8)}
    batch = {
        "image": torch.arange(2 * 8 * 8 * 3, dtype=torch.int64).reshape(2, 8, 8, 3),
        "image_idx": torch.tensor([4, 9], dtype=torch.long),
    }
    return sampler, batch


def _prefetcher(sampler, batch):
    snapshot = sampler.build_prefetch_snapshot(batch)
    signature = sampler.prefetch_live_signature(batch)
    return DeterministicCPUBatchPrefetcher(
        sample_batch=snapshot.sample,
        fallback_sample_batch=lambda: sampler.sample(batch),
        get_sample_count=lambda: sampler.sample_count,
        commit_sample_count=lambda value: setattr(sampler, "sample_count", value),
        get_signature=lambda: sampler.prefetch_live_signature(batch),
        supported_signature=signature,
    )


@pytest.mark.parametrize("sample_count", [0, 1, 3, 7])
def test_private_generator_snapshot_is_byte_exact_to_sync_fas(sample_count: int) -> None:
    sampler, batch = _sampler_and_batch()
    snapshot = sampler.build_prefetch_snapshot(batch)

    sampler.sample_count = sample_count
    torch.manual_seed(4919)
    expected = sampler.sample(batch)
    expected_rng = torch.get_rng_state().clone()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(4919)
    actual = snapshot.sample(generator, sample_count)
    assert torch.equal(actual["indices"], expected["indices"])
    assert torch.equal(actual["image"], expected["image"])
    assert torch.equal(generator.get_state(), expected_rng)


def test_prefetch_matches_sync_across_cpu_rng_invalidation_and_clean_commit() -> None:
    expected_sampler, expected_batch = _sampler_and_batch()
    torch.manual_seed(1701)
    expected_batches = [expected_sampler.sample(expected_batch)]
    expected_eval_draw = torch.rand(11)
    expected_batches.append(expected_sampler.sample(expected_batch))
    expected_batches.append(expected_sampler.sample(expected_batch))
    expected_rng = torch.get_rng_state().clone()

    sampler, batch = _sampler_and_batch()
    torch.manual_seed(1701)
    prefetcher = _prefetcher(sampler, batch)
    actual_batches = [prefetcher.next_batch(0)]
    # This models an eval/grid CPU RNG callback.  The queued batch is now stale
    # and must fall back synchronously without any explicit barrier.
    actual_eval_draw = torch.rand(11)
    actual_batches.append(prefetcher.next_batch(1))
    # No intervening CPU RNG consumer: the next queued batch commits directly.
    actual_batches.append(prefetcher.next_batch(2))
    prefetcher.close()

    assert torch.equal(actual_eval_draw, expected_eval_draw)
    for actual, expected in zip(actual_batches, expected_batches):
        assert torch.equal(actual["indices"], expected["indices"])
        assert torch.equal(actual["image"], expected["image"])
    assert sampler.sample_count == expected_sampler.sample_count == 3
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert prefetcher.discard_count == 2  # one invalid transaction plus final queue cleanup


def test_seeded_prefetch_is_step_addressed_and_never_touches_global_rng() -> None:
    expected_sampler, expected_batch = _sampler_and_batch()
    base_seed = 42
    seeds = [stream_seed(base_seed, "pixel", step) for step in range(3)]
    expected_batches = []
    for step in range(2):
        with fork_seeded_rng(base_seed, "pixel", step, "cpu"):
            expected_batches.append(expected_sampler.sample(expected_batch))

    sampler, batch = _sampler_and_batch()
    torch.manual_seed(9182)
    global_before = torch.get_rng_state().clone()
    prefetcher = _prefetcher(sampler, batch)
    actual_batches = [
        prefetcher.next_batch_seeded(0, seeds[0], seeds[1]),
        prefetcher.next_batch_seeded(1, seeds[1], seeds[2]),
    ]

    assert sampler.sample_count == 2
    assert expected_sampler.sample_count == 2
    assert torch.equal(torch.get_rng_state(), global_before)
    for actual, expected in zip(actual_batches, expected_batches):
        assert torch.equal(actual["indices"], expected["indices"])
        assert torch.equal(actual["image"], expected["image"])
    assert prefetcher.discard_count == 0
    prefetcher.close()


def test_seeded_prefetch_ignores_unrelated_global_cpu_draws() -> None:
    reference_sampler, reference_batch = _sampler_and_batch()
    reference_prefetcher = _prefetcher(reference_sampler, reference_batch)
    reference_first = reference_prefetcher.next_batch_seeded(0, 7001, 7002)
    reference_second = reference_prefetcher.next_batch_seeded(1, 7002, 7003)
    reference_prefetcher.close()

    sampler, batch = _sampler_and_batch()
    torch.manual_seed(8128)
    prefetcher = _prefetcher(sampler, batch)
    actual_first = prefetcher.next_batch_seeded(0, 7001, 7002)
    torch.rand(97)
    global_after_draw = torch.get_rng_state().clone()
    actual_second = prefetcher.next_batch_seeded(1, 7002, 7003)

    assert torch.equal(actual_first["indices"], reference_first["indices"])
    assert torch.equal(actual_second["indices"], reference_second["indices"])
    assert torch.equal(torch.get_rng_state(), global_after_draw)
    assert prefetcher.discard_count == 0
    prefetcher.close()


def test_seeded_prefetch_rederives_requested_step_after_stale_target() -> None:
    expected_sampler, expected_batch = _sampler_and_batch()
    expected_sampler.sample_count = 2
    snapshot = expected_sampler.build_prefetch_snapshot(expected_batch)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(9902)
    expected = snapshot.sample(generator, 2)

    sampler, batch = _sampler_and_batch()
    torch.manual_seed(7711)
    global_before = torch.get_rng_state().clone()
    prefetcher = _prefetcher(sampler, batch)
    prefetcher.next_batch_seeded(0, 9900, 9901)
    sampler.sample_count = 2
    actual = prefetcher.next_batch_seeded(2, 9902, 9903)

    assert torch.equal(actual["indices"], expected["indices"])
    assert torch.equal(actual["image"], expected["image"])
    assert sampler.sample_count == 3
    assert torch.equal(torch.get_rng_state(), global_before)
    assert prefetcher.discard_count == 1
    prefetcher.close()


def test_seeded_prefetch_signature_change_fails_closed_without_global_rng_mutation() -> None:
    sampler, batch = _sampler_and_batch()
    torch.manual_seed(7712)
    global_before = torch.get_rng_state().clone()
    prefetcher = _prefetcher(sampler, batch)
    prefetcher.next_batch_seeded(0, 8800, 8801)
    sampler.config.fas_strength = 0.25

    with pytest.raises(RuntimeError, match="supported sampling signature"):
        prefetcher.next_batch_seeded(1, 8801, 8802)

    assert sampler.sample_count == 1
    assert torch.equal(torch.get_rng_state(), global_before)
    prefetcher.close()


def test_discard_pending_does_not_commit_rng_or_sample_count() -> None:
    sampler, batch = _sampler_and_batch()
    torch.manual_seed(8821)
    prefetcher = _prefetcher(sampler, batch)
    prefetcher.next_batch(0)
    rng_after_first = torch.get_rng_state().clone()
    assert sampler.sample_count == 1

    prefetcher.discard_pending()
    assert sampler.sample_count == 1
    assert torch.equal(torch.get_rng_state(), rng_after_first)

    expected_sampler, expected_batch = _sampler_and_batch()
    expected_sampler.sample_count = 1
    torch.set_rng_state(rng_after_first)
    expected = expected_sampler.sample(expected_batch)
    expected_rng = torch.get_rng_state().clone()

    sampler.sample_count = 1
    torch.set_rng_state(rng_after_first)
    actual = prefetcher.next_batch(1)
    prefetcher.close()
    assert torch.equal(actual["indices"], expected["indices"])
    assert torch.equal(actual["image"], expected["image"])
    assert sampler.sample_count == 2
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_worker_exception_leaves_global_rng_and_count_unmodified() -> None:
    count = [4]
    signature = (4096, "fixed")
    entered = threading.Event()

    def raise_after_private_draw(generator: torch.Generator, _sample_count: int):
        torch.rand(8, generator=generator)
        entered.set()
        raise RuntimeError("prefetch failure")

    prefetcher = DeterministicCPUBatchPrefetcher(
        sample_batch=raise_after_private_draw,
        fallback_sample_batch=lambda: {},
        get_sample_count=lambda: count[0],
        commit_sample_count=lambda value: count.__setitem__(0, value),
        get_signature=lambda: signature,
        supported_signature=signature,
    )
    torch.manual_seed(2718)
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="prefetch failure"):
        prefetcher.next_batch(12)
    assert entered.is_set()
    assert count[0] == 4
    assert torch.equal(torch.get_rng_state(), rng_before)
    prefetcher.close()


def test_signature_change_discards_stale_snapshot_and_stays_synchronous() -> None:
    sampler, batch = _sampler_and_batch()
    torch.manual_seed(31415)
    prefetcher = _prefetcher(sampler, batch)
    prefetcher.next_batch(0)
    assert prefetcher.has_pending_batch

    sampler.config.fas_strength = 0.25
    state_before = torch.get_rng_state().clone()
    count_before = sampler.sample_count
    expected_sampler, expected_batch = _sampler_and_batch(fas_strength=0.25)
    expected_sampler.sample_count = count_before
    torch.set_rng_state(state_before)
    expected = expected_sampler.sample(expected_batch)
    expected_rng = torch.get_rng_state().clone()

    sampler.sample_count = count_before
    torch.set_rng_state(state_before)
    actual = prefetcher.next_batch(1)
    assert torch.equal(actual["indices"], expected["indices"])
    assert torch.equal(actual["image"], expected["image"])
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert sampler.sample_count == count_before + 1
    assert not prefetcher.has_pending_batch
    prefetcher.close()


@pytest.mark.parametrize("requested_step,external_count_delta", [(2, 0), (1, 1)])
def test_target_step_or_sample_count_mismatch_falls_back_exactly(
    requested_step: int, external_count_delta: int
) -> None:
    sampler, batch = _sampler_and_batch()
    torch.manual_seed(16180)
    prefetcher = _prefetcher(sampler, batch)
    prefetcher.next_batch(0)
    sampler.sample_count += external_count_delta
    state_before = torch.get_rng_state().clone()
    count_before = sampler.sample_count

    expected_sampler, expected_batch = _sampler_and_batch()
    expected_sampler.sample_count = count_before
    torch.set_rng_state(state_before)
    expected = expected_sampler.sample(expected_batch)
    expected_rng = torch.get_rng_state().clone()

    sampler.sample_count = count_before
    torch.set_rng_state(state_before)
    actual = prefetcher.next_batch(requested_step)
    prefetcher.close()
    assert torch.equal(actual["indices"], expected["indices"])
    assert torch.equal(actual["image"], expected["image"])
    assert sampler.sample_count == count_before + 1
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_live_signature_tracks_num_rays_config_image_order_and_versions() -> None:
    sampler, batch = _sampler_and_batch()
    original = sampler.prefetch_live_signature(batch)
    sampler.set_num_rays_per_batch(65)
    assert sampler.prefetch_live_signature(batch) != original
    sampler.set_num_rays_per_batch(64)
    sampler.config.fas_strength = 0.5
    assert sampler.prefetch_live_signature(batch) != original
    sampler.config.fas_strength = 0.75
    batch["image_idx"][0] = 5
    assert sampler.prefetch_live_signature(batch) != original
    batch["image_idx"][0] = 4
    batch["image"].add_(1)
    assert sampler.prefetch_live_signature(batch) != original


@pytest.mark.parametrize("key", ["image", "image_idx"])
def test_live_signature_tracks_tensor_replacement_without_reading_values(key: str) -> None:
    sampler, batch = _sampler_and_batch()
    original = sampler.prefetch_live_signature(batch)
    batch[key] = batch[key].clone()

    assert sampler.prefetch_live_signature(batch) != original
