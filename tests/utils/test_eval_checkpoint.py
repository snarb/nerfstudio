"""Exact checkpoint selection during evaluation."""

from types import SimpleNamespace

import torch

from nerfstudio.utils.eval_utils import eval_load_checkpoint


def test_eval_load_checkpoint_prefers_exact_override(tmp_path):
    exact = tmp_path / "arbitrary-name.ckpt"
    torch.save({"step": 123, "pipeline": {"weight": torch.tensor([4.0])}}, exact)
    pipeline = SimpleNamespace(load_pipeline=lambda state, step: setattr(pipeline, "loaded", (state, step)))
    config = SimpleNamespace(eval_checkpoint=exact, load_dir=tmp_path / "missing", load_step=None)

    loaded_path, loaded_step = eval_load_checkpoint(config, pipeline)

    assert loaded_path == exact
    assert loaded_step == 123
    assert pipeline.loaded[1] == 123
    assert torch.equal(pipeline.loaded[0]["weight"], torch.tensor([4.0]))


def test_eval_load_checkpoint_keeps_latest_run_selection_without_override(tmp_path):
    latest = tmp_path / "step-000000007.ckpt"
    torch.save({"step": 7, "pipeline": {"weight": torch.tensor([2.0])}}, latest)
    pipeline = SimpleNamespace(load_pipeline=lambda state, step: setattr(pipeline, "loaded", (state, step)))
    config = SimpleNamespace(eval_checkpoint=None, load_dir=tmp_path, load_step=None)

    loaded_path, loaded_step = eval_load_checkpoint(config, pipeline)

    assert loaded_path == latest
    assert loaded_step == 7
    assert pipeline.loaded[1] == 7
    assert torch.equal(pipeline.loaded[0]["weight"], torch.tensor([2.0]))
