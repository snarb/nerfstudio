from __future__ import annotations

import copy

import pytest
import torch

from nerfstudio.engine.optimizers import AdamOptimizerConfig, Optimizers


def _optimizers(parameter: torch.nn.Parameter, fused: bool | None) -> Optimizers:
    return Optimizers(
        {"fields": {"optimizer": AdamOptimizerConfig(lr=1e-2, fused=fused), "scheduler": None}},
        {"fields": [parameter]},
    )


def test_fused_adam_is_opt_in() -> None:
    parameter = torch.nn.Parameter(torch.ones(4))
    optimizer = _optimizers(parameter, fused=None).optimizers["fields"]
    assert optimizer.param_groups[0]["fused"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused Adam")
def test_fused_adam_loads_foreach_state_and_matches_one_scaled_step() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    parent_parameter = torch.nn.Parameter(torch.ones(4096, device=device))
    parent = _optimizers(parent_parameter, fused=None)
    parent_parameter.grad = torch.linspace(-1.0, 1.0, parent_parameter.numel(), device=device)
    parent.optimizers["fields"].step()
    loaded_state = {"fields": copy.deepcopy(parent.optimizers["fields"].state_dict())}

    outputs: list[torch.Tensor] = []
    states: list[dict[str, torch.Tensor]] = []
    for fused in (None, True):
        parameter = torch.nn.Parameter(parent_parameter.detach().clone())
        optimizers = _optimizers(parameter, fused=fused)
        optimizers.load_optimizers(copy.deepcopy(loaded_state))
        optimizer = optimizers.optimizers["fields"]
        if fused:
            assert optimizer.param_groups[0]["fused"] is True
            assert optimizer.state[parameter]["step"].device == device

        scaler = torch.amp.GradScaler("cuda", init_scale=8192.0)
        gradient = torch.cos(torch.arange(parameter.numel(), device=device, dtype=torch.float32))
        scaler.scale((parameter * gradient).sum()).backward()
        scaler.step(optimizer)
        scaler.update()
        outputs.append(parameter.detach().clone())
        states.append(optimizer.state[parameter])

    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.0, atol=1.2e-7)
    torch.testing.assert_close(states[0]["exp_avg"], states[1]["exp_avg"], rtol=0.0, atol=1e-7)
    torch.testing.assert_close(states[0]["exp_avg_sq"], states[1]["exp_avg_sq"], rtol=0.0, atol=1e-7)

    live_parameter = torch.nn.Parameter(parent_parameter.detach().clone())
    live_optimizers = _optimizers(live_parameter, fused=None)
    live_optimizers.load_optimizers(copy.deepcopy(loaded_state))
    live_optimizers.set_adam_fused(True, param_group_names=["fields"])
    live_scaler = torch.amp.GradScaler("cuda", init_scale=8192.0)
    live_gradient = torch.cos(
        torch.arange(live_parameter.numel(), device=device, dtype=torch.float32)
    )
    live_scaler.scale((live_parameter * live_gradient).sum()).backward()
    live_scaler.step(live_optimizers.optimizers["fields"])
    live_scaler.update()
    live_state = live_optimizers.optimizers["fields"].state[live_parameter]
    torch.testing.assert_close(outputs[1], live_parameter, rtol=0.0, atol=0.0)
    torch.testing.assert_close(states[1]["exp_avg"], live_state["exp_avg"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(states[1]["exp_avg_sq"], live_state["exp_avg_sq"], rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused Adam")
def test_live_fused_adam_switch_preserves_optimizer_and_moments() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    parameter = torch.nn.Parameter(torch.ones(4096, device=device))
    optimizers = _optimizers(parameter, fused=None)
    optimizer = optimizers.optimizers["fields"]
    parameter.grad = torch.linspace(-1.0, 1.0, parameter.numel(), device=device)
    optimizer.step()

    optimizer_id = id(optimizer)
    parameter_id = id(parameter)
    exp_avg = optimizer.state[parameter]["exp_avg"]
    exp_avg_sq = optimizer.state[parameter]["exp_avg_sq"]
    exp_avg_before = exp_avg.clone()
    exp_avg_sq_before = exp_avg_sq.clone()
    step_before = optimizer.state[parameter]["step"].clone()

    optimizers.set_adam_fused(True, param_group_names=["fields"])

    assert id(optimizers.optimizers["fields"]) == optimizer_id
    assert id(optimizer.param_groups[0]["params"][0]) == parameter_id
    assert id(optimizer.state[parameter]["exp_avg"]) == id(exp_avg)
    assert id(optimizer.state[parameter]["exp_avg_sq"]) == id(exp_avg_sq)
    assert optimizer.param_groups[0]["fused"] is True
    assert optimizer.defaults["fused"] is True
    assert optimizer._step_supports_amp_scaling is True
    assert optimizer.state[parameter]["step"].device == device
    torch.testing.assert_close(optimizer.state[parameter]["step"].cpu(), step_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(exp_avg, exp_avg_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(exp_avg_sq, exp_avg_sq_before, rtol=0.0, atol=0.0)


def test_live_fused_adam_switch_rejects_cpu_before_mutation() -> None:
    parameter = torch.nn.Parameter(torch.ones(4))
    optimizers = _optimizers(parameter, fused=None)
    optimizer = optimizers.optimizers["fields"]

    with pytest.raises(RuntimeError, match="requires CUDA"):
        optimizers.set_adam_fused(True, param_group_names=["fields"])

    assert optimizer.param_groups[0]["fused"] is None
    assert optimizer.defaults["fused"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused Adam")
def test_live_fused_adam_switch_supports_grad_scaler_and_post_switch_reload() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    parameter = torch.nn.Parameter(torch.ones(4096, device=device))
    live = _optimizers(parameter, fused=None)

    # Create historical Adam moments, then switch without reconstructing.
    parameter.grad = torch.linspace(-1.0, 1.0, parameter.numel(), device=device)
    live.optimizers["fields"].step()
    live.set_adam_fused(True, param_group_names=["fields"])
    scaler = torch.amp.GradScaler("cuda", init_scale=8192.0)
    scaler.scale(parameter.square().sum()).backward()
    scaler.step(live.optimizers["fields"])
    scaler.update()

    saved_parameter = parameter.detach().clone()
    saved_optimizer = copy.deepcopy(live.optimizers["fields"].state_dict())
    saved_scaler = copy.deepcopy(scaler.state_dict())

    reloaded_parameter = torch.nn.Parameter(saved_parameter.clone())
    reloaded = _optimizers(reloaded_parameter, fused=None)
    reloaded.load_optimizers({"fields": copy.deepcopy(saved_optimizer)})
    reloaded_optimizer = reloaded.optimizers["fields"]
    assert reloaded_optimizer.param_groups[0]["fused"] is True
    assert reloaded_optimizer._step_supports_amp_scaling is True
    assert reloaded_optimizer.state[reloaded_parameter]["step"].device == device

    reloaded_scaler = torch.amp.GradScaler("cuda")
    reloaded_scaler.load_state_dict(saved_scaler)
    reloaded_scaler.scale(reloaded_parameter.square().sum()).backward()
    reloaded_scaler.step(reloaded_optimizer)
    reloaded_scaler.update()
    assert torch.isfinite(reloaded_parameter).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fused Adam")
def test_live_fused_adam_switch_matches_initialized_fused_inf_skip() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    parent_parameter = torch.nn.Parameter(torch.ones(256, device=device))
    parent = _optimizers(parent_parameter, fused=None)
    parent_parameter.grad = torch.ones_like(parent_parameter)
    parent.optimizers["fields"].step()
    parent_state = copy.deepcopy(parent.optimizers["fields"].state_dict())

    results = []
    for mode in ("initialized", "live"):
        parameter = torch.nn.Parameter(parent_parameter.detach().clone())
        optimizers = _optimizers(parameter, fused=True if mode == "initialized" else None)
        optimizers.load_optimizers({"fields": copy.deepcopy(parent_state)})
        if mode == "live":
            optimizers.set_adam_fused(True, param_group_names=["fields"])
        optimizer = optimizers.optimizers["fields"]
        before_parameter = parameter.detach().clone()
        before_state = {
            key: value.detach().clone() for key, value in optimizer.state[parameter].items()
        }
        scaler = torch.amp.GradScaler("cuda", init_scale=8192.0)
        scaler.scale((parameter * torch.full_like(parameter, float("inf"))).sum()).backward()
        scaler.step(optimizer)
        scaler.update()
        results.append(
            (
                parameter.detach().clone(),
                {key: value.detach().clone() for key, value in optimizer.state[parameter].items()},
                scaler.get_scale(),
            )
        )
        torch.testing.assert_close(parameter, before_parameter, rtol=0.0, atol=0.0)
        for key, value in before_state.items():
            torch.testing.assert_close(optimizer.state[parameter][key], value, rtol=0.0, atol=0.0)

    torch.testing.assert_close(results[0][0], results[1][0], rtol=0.0, atol=0.0)
    for key in results[0][1]:
        torch.testing.assert_close(results[0][1][key], results[1][1][key], rtol=0.0, atol=0.0)
    assert results[0][2] == results[1][2] == 4096.0
