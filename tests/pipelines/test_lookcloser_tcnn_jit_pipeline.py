"""CPU-only live-switch and checkpoint-resync tests for scoped TCNN JIT."""

from types import SimpleNamespace

import pytest
from torch import nn

from nerfstudio.pipelines.lookcloser_pipeline import LookCloserPipeline


class _FakeField:
    def __init__(self, *, geometry: bool = False, color: bool = False) -> None:
        self.states = {"geometry": geometry, "color": color}
        self.set_calls = []

    @staticmethod
    def _selected(scope: str):
        return ("geometry", "color") if scope == "both" else (scope,)

    def set_tcnn_network_jit(self, enabled: bool, scope: str = "both") -> None:
        self.set_calls.append((enabled, scope))
        for name in self._selected(scope):
            self.states[name] = enabled

    def get_tcnn_network_jit(self, scope: str = "both") -> bool:
        states = [self.states[name] for name in self._selected(scope)]
        if any(state != states[0] for state in states[1:]):
            raise RuntimeError("states disagree")
        return states[0]


def _pipeline(
    *,
    scope: str,
    switch_step: int | None,
    field: _FakeField,
    second_step: int | None = None,
    second_scope: str | None = None,
) -> LookCloserPipeline:
    pipeline = LookCloserPipeline.__new__(LookCloserPipeline)
    nn.Module.__init__(pipeline)
    pipeline.config = SimpleNamespace(
        tcnn_network_jit_switch_step=switch_step,
        tcnn_network_jit_second_switch_step=second_step,
        tcnn_network_jit_second_switch_scope=second_scope,
        model=SimpleNamespace(tcnn_network_jit=False, tcnn_network_jit_scope=scope),
    )
    pipeline._model = SimpleNamespace(field=field)
    pipeline._tcnn_network_jit_switch_applied = False
    pipeline._tcnn_network_jit_second_switch_applied = False
    return pipeline


@pytest.mark.parametrize("scope", ["both", "geometry", "color"])
def test_live_switch_obeys_boundary_and_is_idempotent(scope: str) -> None:
    field = _FakeField()
    pipeline = _pipeline(scope=scope, switch_step=10, field=field)

    pipeline._apply_live_tcnn_jit_switch(9)
    assert field.set_calls == []

    pipeline._apply_live_tcnn_jit_switch(10)
    pipeline._apply_live_tcnn_jit_switch(11)

    assert field.set_calls == [(True, scope)]
    selected = _FakeField._selected(scope)
    assert all(field.states[name] for name in selected)
    assert all(not field.states[name] for name in {"geometry", "color"} - set(selected))


def test_checkpoint_resync_restores_exact_per_network_state() -> None:
    field = _FakeField(geometry=True, color=True)
    pipeline = _pipeline(scope="color", switch_step=10, field=field)

    pipeline._sync_tcnn_network_jit_to_step(9)
    assert field.states == {"geometry": False, "color": False}
    assert pipeline._tcnn_network_jit_switch_applied is False

    pipeline._sync_tcnn_network_jit_to_step(10)
    assert field.states == {"geometry": False, "color": True}
    assert pipeline._tcnn_network_jit_switch_applied is True


def test_live_switch_asserts_if_selected_state_is_lost() -> None:
    field = _FakeField()
    pipeline = _pipeline(scope="color", switch_step=10, field=field)
    pipeline._tcnn_network_jit_switch_applied = True

    with pytest.raises(RuntimeError, match="live switch.*color"):
        pipeline._apply_live_tcnn_jit_switch(11)


def test_second_switch_enables_additional_scope_at_strictly_later_boundary() -> None:
    field = _FakeField()
    pipeline = _pipeline(
        scope="geometry",
        switch_step=10,
        field=field,
        second_step=20,
        second_scope="color",
    )

    pipeline._apply_live_tcnn_jit_switch(9)
    pipeline._apply_live_tcnn_jit_switch(10)
    pipeline._apply_live_tcnn_jit_switch(19)
    assert field.states == {"geometry": True, "color": False}

    pipeline._apply_live_tcnn_jit_switch(20)
    pipeline._apply_live_tcnn_jit_switch(21)
    assert field.states == {"geometry": True, "color": True}
    assert field.set_calls == [(True, "geometry"), (True, "color")]
    assert pipeline._tcnn_network_jit_switch_applied is True
    assert pipeline._tcnn_network_jit_second_switch_applied is True


def test_second_switch_is_recorded_even_when_first_scope_already_enabled_it() -> None:
    field = _FakeField()
    pipeline = _pipeline(
        scope="both",
        switch_step=10,
        field=field,
        second_step=20,
        second_scope="color",
    )

    pipeline._apply_live_tcnn_jit_switch(20)
    pipeline._apply_live_tcnn_jit_switch(21)

    assert field.set_calls == [(True, "both"), (True, "color")]
    assert pipeline._tcnn_network_jit_second_switch_applied is True


@pytest.mark.parametrize(
    ("switch_step", "second_step", "second_scope", "error"),
    [
        (10, 20, None, "must be set together"),
        (10, None, "color", "must be set together"),
        (None, 20, "color", "requires the first"),
        (10, 10, "color", "strictly greater"),
        (10, 9, "color", "strictly greater"),
        (10, 20, "invalid", "Unsupported TCNN network JIT scope"),
    ],
)
def test_invalid_second_switch_schedules_fail_closed(
    switch_step: int | None,
    second_step: int | None,
    second_scope: str | None,
    error: str,
) -> None:
    pipeline = _pipeline(
        scope="geometry",
        switch_step=switch_step,
        field=_FakeField(),
        second_step=second_step,
        second_scope=second_scope,
    )

    with pytest.raises(ValueError, match=error):
        pipeline._validate_tcnn_network_jit_switch_config()


@pytest.mark.parametrize("step", [0, 9, 10, 19, 20, 30])
def test_checkpoint_resync_reconstructs_both_switches(step: int) -> None:
    field = _FakeField(geometry=True, color=True)
    pipeline = _pipeline(
        scope="geometry",
        switch_step=10,
        field=field,
        second_step=20,
        second_scope="color",
    )

    pipeline._sync_tcnn_network_jit_to_step(step)

    assert field.states == {
        "geometry": step >= 10,
        "color": step >= 20,
    }
    assert pipeline._tcnn_network_jit_switch_applied is (step >= 10)
    assert pipeline._tcnn_network_jit_second_switch_applied is (step >= 20)
