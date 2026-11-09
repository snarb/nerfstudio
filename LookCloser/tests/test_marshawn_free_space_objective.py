import json,sys
from pathlib import Path
import numpy as np
import pytest
import torch

scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
from marshawn_free_space_objective import FreeSpaceObjective
from probe_marshawn_occupancy import file_sha


def artifact(tmp_path):
    origin=np.zeros((9,3));direction=np.tile([1.,0,0],(9,1));start=np.full(9,.01);end=np.linspace(.2,.8,9);distance=np.ones(9)
    path=tmp_path/'rays.npz';np.savez(path,origins=origin,directions=direction,t_start=start,t_end=end,witness_depth=distance,surface_margin=np.full(9,.05),camera_idx=np.repeat([2,5,9],[1,3,5]),aabb=[[-1,-1,-1],[1,1,1]])
    transforms=tmp_path/'transforms.json';transforms.write_text('{}');preparation=tmp_path/'preparation.json';preparation.write_text('{}')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(kind='observed_train_sfm_free_ray_interval_candidates',coordinate_system='prepared_world',uses_eval_rgb=False,uses_masks=False,bindings=dict(transforms_sha256=file_sha(transforms),normalization_sha256=file_sha(preparation)),policy={},variants=[dict(name='scene_tight',file=path.name,sha256=file_sha(path),aabb=[[-1,-1,-1],[1,1,1]])])))
    return dict(manifest=manifest,variant='scene_tight',current_bounds=[[-1,-1,-1],[1,1,1]],transforms=transforms,preparation=preparation)


class Constant(torch.nn.Module):
    def __init__(self,value=100.):super().__init__();self.sigma=torch.nn.Parameter(torch.tensor(value))
    def density_fn(self,points):return self.sigma.expand(len(points),1)


def test_constant_density_optical_integral_gradient_does_not_saturate(tmp_path):
    objective=FreeSpaceObjective(**artifact(tmp_path),rays=128,samples=64);field=Constant();batch=objective.sample_batch(7);loss,metrics=objective(field,7)
    expected=100*batch['length'].mean()
    torch.testing.assert_close(metrics['free_tau_mean'],expected)
    loss.backward();torch.testing.assert_close(field.sigma.grad,.01*batch['length'].mean())
    assert field.sigma.grad>0 and all(not v.requires_grad for v in metrics.values())
    assert not list(objective.parameters()) and not objective.state_dict()


def test_camera_balance_private_rng_and_stratification(tmp_path):
    objective=FreeSpaceObjective(**artifact(tmp_path));before=torch.random.get_rng_state();a=objective.sample_batch(20);after=torch.random.get_rng_state();b=objective.sample_batch(20);c=objective.sample_batch(21)
    assert torch.equal(before,after) and torch.equal(a['points'],b['points']) and not torch.equal(a['points'],c['points'])
    counts=a['camera_idx'].unique(return_counts=True)[1];assert int(counts.max()-counts.min())<=1
    start=objective.starts[a['indices']];unit=(a['t']-start[:,None])/a['length'][:,None];bins=(unit*64).floor().long();torch.testing.assert_close(bins,torch.arange(64).expand(128,-1))


@pytest.mark.parametrize('change',['bounds','transforms','normalization','archive'])
def test_mismatched_artifact_rejected(tmp_path,change):
    args=artifact(tmp_path)
    if change=='bounds':args['current_bounds']=[[-2,-1,-1],[1,1,1]]
    elif change=='transforms':args['transforms'].write_text('{"changed":true}')
    elif change=='normalization':args['preparation'].write_text('{"changed":true}')
    else:(tmp_path/'rays.npz').write_bytes(b'changed')
    with pytest.raises(ValueError):FreeSpaceObjective(**args)
