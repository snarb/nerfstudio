from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.expand_lookcloser_hash_checkpoint import (
    ENCODING_KEY,
    aligned_level_rows,
    expand_checkpoint_state,
    expand_hash_tensor,
    expected_parameter_count,
)


def test_real_leader_and_hash24_parameter_counts_match_tcnn_layout() -> None:
    common = dict(num_levels=16, min_res=16.0, max_res=8192.0, features_per_level=2)
    assert expected_parameter_count(log2_hashmap_size=23, **common) == 171_739_264
    assert expected_parameter_count(log2_hashmap_size=24, **common) == 322_734_208


def test_expansion_repeats_each_saturated_level_partition() -> None:
    common = dict(num_levels=3, min_res=2.0, max_res=32.0, features_per_level=2)
    grid_shape = {key: common[key] for key in ("num_levels", "min_res", "max_res")}
    source_rows = aligned_level_rows(log2_hashmap_size=4, **grid_shape)
    source = torch.arange(sum(source_rows) * 2, dtype=torch.float32)
    target = expand_hash_tensor(source, source_log2=4, target_log2=5, **common)
    target_rows = aligned_level_rows(log2_hashmap_size=5, **grid_shape)

    source_offset = 0
    target_offset = 0
    for old_rows, new_rows in zip(source_rows, target_rows):
        old_count = old_rows * 2
        new_count = new_rows * 2
        old_table = source[source_offset : source_offset + old_count].reshape(old_rows, 2)
        new_table = target[target_offset : target_offset + new_count].reshape(new_rows, 2)
        assert torch.equal(new_table, old_table.repeat((new_rows // old_rows, 1)))
        source_offset += old_count
        target_offset += new_count


def test_checkpoint_expands_encoding_and_matching_adam_moments() -> None:
    common = dict(num_levels=3, min_res=2.0, max_res=32.0, features_per_level=2)
    source_count = expected_parameter_count(log2_hashmap_size=4, **common)
    encoding = torch.arange(source_count, dtype=torch.float32)
    checkpoint = {
        "pipeline": {ENCODING_KEY: encoding.clone()},
        "optimizers": {
            "fields": {
                "state": {
                    0: {
                        "step": torch.tensor(10.0),
                        "exp_avg": encoding.clone(),
                        "exp_avg_sq": encoding.clone() + 1,
                    },
                    1: {
                        "step": torch.tensor(10.0),
                        "exp_avg": torch.zeros(7),
                        "exp_avg_sq": torch.ones(7),
                    },
                },
                "param_groups": [{"params": [0, 1]}],
            }
        },
    }
    audit = expand_checkpoint_state(
        checkpoint, source_log2=4, target_log2=5, **common
    )
    target_count = expected_parameter_count(log2_hashmap_size=5, **common)
    assert checkpoint["pipeline"][ENCODING_KEY].numel() == target_count
    assert checkpoint["optimizers"]["fields"]["state"][0]["exp_avg"].numel() == target_count
    assert checkpoint["optimizers"]["fields"]["state"][0]["exp_avg_sq"].numel() == target_count
    assert checkpoint["optimizers"]["fields"]["state"][1]["exp_avg"].numel() == 7
    assert audit["expanded_optimizer_moments"] == ["0.exp_avg", "0.exp_avg_sq"]
