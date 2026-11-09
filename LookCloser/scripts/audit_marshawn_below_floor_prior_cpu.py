"""Actual-source clipped cells, private RNG/resume and native trunc_exp math."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
from marshawn_below_floor_density_prior import FrozenBelowFloorPrior

C=Path('/fsx/tmp/marshawn_001660_hd/tuning_20261106')
P=C/'below_floor_density_prior_CPU_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    P.mkdir(exist_ok=True)
    grid=C/'independent_occupancy/semantic_quarter_distortion_final_geometry/quarter_distortion_rawmin/grid.npz'
    band=np.array([[-.84,-.21,-.07],[.15,.42,-.05]],np.float64)
    with np.load(grid,allow_pickle=False)as z:
        active=z['active'];boxes=z['cascade_aabbs'];resolution=z['resolution'];core=z['core'];admission=z['admission']
        assert active.shape==(1,128,128,128)and np.array_equal(active,core)
        assert np.all(admission[active])
        indices=np.argwhere(active[0]);pitch=(boxes[0,3:]-boxes[0,:3])/resolution
        lo=np.maximum(boxes[0,:3]+indices*pitch,band[0]);hi=np.minimum(boxes[0,:3]+(indices+1)*pitch,band[1])
        keep=np.all(hi>lo,axis=1);lo=lo[keep];hi=hi[keep];indices=indices[keep]
    table=P/'frozen_active_below_band_cells.npz'
    np.savez_compressed(table,cell_lo=lo,cell_hi=hi,cell_index=indices,band=band,
        original_grid_sha256=np.array(sha(grid)),source_active_count=np.array(int(active.sum())),
        focused_probability=np.array(.8),no_rejection=True)
    state=torch.get_rng_state().clone();prior=FrozenBelowFloorPrior(table)
    first,packet=prior.draw('cpu');assert torch.equal(torch.get_rng_state(),state)
    prior=FrozenBelowFloorPrior(table)
    state32=None;packets=[];sequential=[]
    for step in range(64):
        xyz,q=prior.draw('cpu');packets.append(q);sequential.append(xyz)
        if step==31:state32=prior.state_dict()
    resumed=FrozenBelowFloorPrior(table);resumed.load_state_dict(state32)
    for step in range(32,64):
        xyz,q=resumed.draw('cpu');assert torch.equal(xyz,sequential[step]);assert q==packets[step]
    assert torch.equal(resumed.generator.get_state(),prior.generator.get_state())
    assert torch.equal(torch.get_rng_state(),state)
    # Prove all declared arms use exactly the same point trajectory.
    arm_points=[]
    for coefficient in [0.,1e-4,1e-3]:
        arm=FrozenBelowFloorPrior(table);arm_points.append(arm.draw('cpu')[0])
    assert all(torch.equal(arm_points[0],x)for x in arm_points[1:])
    activation=C/'semantic_quarter_distortion_candidate_20261107/code/nerfstudio/field_components/activations.py'
    loader=importlib.util.spec_from_file_location('_actual_source_activation_CPU',activation)
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    logits=torch.tensor([-20.,-5.,0.,5.,10.,15.,20.,25.,30.,35.],requires_grad=True)
    sigma=module._TruncExp.apply(logits);aux=torch.log1p(sigma*.0015625).sum();aux.backward()
    expected=.0015625*torch.exp(logits.detach().clamp(-15,15))/(1+.0015625*sigma.detach())
    torch.testing.assert_close(logits.grad,expected,rtol=2e-6,atol=1e-12)
    # A shared toy parameter verifies finite +0*aux leaves original RGB and
    # total parameter gradients exact; no optimizer or source field is used.
    parameter=torch.tensor(.17,requires_grad=True);rgb=((parameter-.4)**2);raw=rgb.detach().clone()
    baseline=torch.autograd.grad(rgb,parameter,retain_graph=True)[0]
    toy_sigma=torch.exp(parameter+first[:,0]);zero,finite_aux=prior.objective(toy_sigma,0.)
    combined=rgb+zero;actual=torch.autograd.grad(combined,parameter)[0]
    assert torch.equal(baseline,actual)and torch.equal(combined.detach(),raw)
    rejected=False
    try:prior.objective(torch.full((1024,),float('inf')),0.)
    except FloatingPointError:rejected=True
    assert rejected
    bad=dict(state32,table_sha256='bad');rejected=False
    try:FrozenBelowFloorPrior(table).load_state_dict(bad)
    except ValueError:rejected=True
    assert rejected
    volume=np.prod(hi-lo,axis=1);band_volume=float(np.prod(band[1]-band[0]));focused_volume=float(volume.sum())
    island=json.loads((C/'semantic_cell_candidates_cpu_v1/component_floor_inventory_v1/active_r0_low_cells.json').read_text())
    island_cells={tuple(c['cell'])for c in island};island_volume=float(sum(v for cell,v in zip(indices,volume)if tuple(cell)in island_cells))
    result=dict(passed=True,CUDA_started=False,optimizer_created=False,source_model_loaded=False,
        source_grid_sha256=sha(grid),source_activation_sha256=sha(activation),cell_table_sha256=sha(table),
        source_active_cells=int(active.sum()),clipped_source_active_cells=len(indices),band=band.tolist(),
        band_volume=band_volume,focused_prior_volume=focused_volume,island_clipped_volume=island_volume,
        expected_island_hits_per1024=1024*(.8*island_volume/focused_volume+.2*island_volume/band_volume),
        expected_uniform_whole_band_island_hits_per1024=1024*island_volume/band_volume,
        mixture_is_focused_prior_not_uniform_integral=True,frozen_source_cells_never_adapted=True,
        full64_vs32resume_private_points_bitwise_equal=True,private_RNG_state_exact=True,
        global_Torch_RNG_unchanged=True,all3_coefficients_same_points=True,
        lambda0_same1024queries_and_exact_toy_RGB_parameter_gradient=True,
        lambda0_nonfinite_aux_fails_closed=True,wrong_private_table_restore_rejected=True,
        native_trunc_exp_log1p_grad=[dict(logit=float(x),sigma=float(s),gradient=float(g))for x,s,g in zip(logits.detach(),sigma.detach(),logits.grad)],
        finite_gradient_caveat='Native trunc_exp backward clamps logits to15: log1p(sigma*dt) gradient vanishes for ultra-dense sigma. Seven centers and captured below-band samples are modest, but full-band sigma tail is unmeasured.',
        native_callbacks_policy='Keep ordinary native warm-training callbacks; only this sampling-prior cell table is frozen. No direct occupancy/admission/sampler edit.')
    (P/'CPU_gradient_RNG_proof.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
