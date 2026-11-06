"""Zero-update actual native radius1 ARM proof from protected R0 weights; not R1 training."""
import argparse,json,os
from pathlib import Path

def main(a):
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch,numpy as np
    from marshawn_semantic_admission import sha,AdmissionLookup
    from marshawn_occupancy_audit import dilate_one
    from nerfstudio.utils.eval_utils import eval_setup
    from probe_marshawn_semantic_admission import parameter_hash
    spec=json.loads(a.spec.read_text());root=a.spec.parent;source=Path(spec['semantic_admission']['manifest']).parent
    torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(0).total_memory)
    if spec['radius']!=1 or spec['width']!=64 or spec['semantic_admission']['identity_control']:raise ValueError('Exact R1/sourceW64 production spec required')
    record=json.loads((a.archive/'step-000015188_grid.json').read_text())
    checkpoint=a.archive/'checkpoint.ckpt'
    if sha(checkpoint)!=record['checkpoint_sha256'] or sha(a.archive/'step-000015188_grid.npz')!=record['grid_archive_sha256']:raise ValueError('Protected R0 inputs changed')
    def update(cfg):
        cfg.eval_checkpoint=checkpoint;cfg.pipeline.model.occupancy_dilation_radius=1
        cfg.pipeline.model.semantic_manifest=str(source/'manifest.json');cfg.pipeline.model.semantic_manifest_sha256=spec['semantic_admission']['manifest_sha256']
        return cfg
    _,pipeline,loaded,step=eval_setup(a.archive/'config.yml',test_mode='val',update_config_callback=update)
    model=pipeline.model;model.eval()
    assert step==15188 and model.config.occupancy_dilation_radius==1
    assert model.field.density_activation=='trunc_exp' and model.field.correct_sh_directions is True
    model.field.feature_reweighting_strength=model.config.feature_reweighting_strength=1.
    with np.load(a.archive/'step-000015188_grid.npz')as z:core=z['core'].copy();r0=z['active'].copy();admission=z['admission'].copy()
    torch.testing.assert_close(model.semantic_learned_core.cpu(),torch.from_numpy(core),rtol=0,atol=0)
    np.testing.assert_array_equal(model.semantic_admission.cpu().numpy()[None],admission)
    ema=model.occupancy_grid.occs.clone();before=parameter_hash(model);raw_core=torch.from_numpy(core).to(model.device)
    expected=dilate_one(core)&admission
    with torch.no_grad():
        model.occupancy_grid.binaries.copy_(raw_core);model._postprocess_occupancy_grid(step)
        first=model.occupancy_grid.binaries.clone()
        np.testing.assert_array_equal(first.cpu().numpy(),expected)
        assert torch.equal(model.semantic_learned_core,raw_core) and torch.equal(model.occupancy_grid.occs,ema)
        model.occupancy_grid.binaries.copy_(raw_core);model._postprocess_occupancy_grid(step)
        assert torch.equal(model.occupancy_grid.binaries,first)
    # Native ARM through actual calibrated camera rays, never a CPU field substitute.
    camera=pipeline.datamanager.train_dataset.cameras[0:1].to(model.device)
    ys=torch.linspace(0,int(camera.height[0])-1,64,device=model.device);xs=torch.linspace(0,int(camera.width[0])-1,64,device=model.device)
    y,x=torch.meshgrid(ys,xs,indexing='ij');coords=torch.stack([y.reshape(-1)+.5,x.reshape(-1)+.5],dim=-1)
    ray=camera.generate_rays(0,coords=coords,keep_shape=False)
    with torch.inference_mode(),torch.autocast('cuda'):
        outputs=model(ray)
    for key in ('rgb','accumulation'):
        if not bool(torch.isfinite(outputs[key]).all()):raise FloatingPointError('Nonfinite actual R1 ARM output')
    q=json.loads((source/'TRAIN103_queries.json').read_text());points=np.array([p['witness_xyz']for p in q['pixels']]);raypoints=[]
    for p,r in zip(q['pixels'],q['rays']):
        xyz=np.array(p['witness_xyz']);o,d=np.array(r['origin']),np.array(r['direction']);raypoints.append(o+d*((xyz-o)@d))
    anchors=[r for r in json.loads((source/'witness_receipt.json').read_text())['anchors']if r['name']in ('toe_upper','sfm_6862')]
    bounds=model.field.aabb;points=torch.tensor(np.vstack([points,raypoints,[r['xyz_floor']for r in anchors]]),device=model.device,dtype=torch.float32)
    source_kept=model.semantic_lookup(points);assert bool(source_kept.all())
    old=AdmissionLookup(torch.from_numpy(r0[0]).to(model.device),bounds)(points);new=AdmissionLookup(first[0],bounds)(points)
    assert not bool((old&~new).any())
    assert torch.equal(model.occupancy_grid.binaries,first)and torch.equal(model.semantic_learned_core,raw_core)and torch.equal(model.occupancy_grid.occs,ema)
    after=parameter_hash(model);assert before==after and all(p.grad is None for p in model.parameters())
    report=dict(passed=True,kind='ACTUAL_NATIVE_R1_POSTPROCESS_AND_ARM_ZERO_UPDATE',source_checkpoint_sha256=sha(checkpoint),source_checkpoint_step=step,weights_source='protected semanticR0; never claimed R1-trained',spec_sha256=sha(a.spec),proof_sha256=sha(__file__),fixed_support_manifest_sha256=spec['semantic_admission']['manifest_sha256'],native_radius=1,semantic_source_pad=1,native_radius_distinct_from_source_pad=True,core_cells=int(core.sum()),R0_active_cells=int(r0.sum()),R1_active_cells=int(first.sum()),native_GPU_matches_independent26neighbor_CPU_dilation_and_admission=True,core_unchanged=True,EMA_unchanged=True,parameters_unchanged_sha256=before,same_fresh_core_reproduces_same_non_cumulative_R1=True,actual_ARM_rays=4096,finite_RGB_alpha=True,positive_source_points_retained=int(source_kept.sum()),positive_source_points_total=len(points),native_R0_positive_point_cells=int(old.sum()),native_R1_positive_point_cells=int(new.sum()),no_existing_positive_native_cell_lost=True,optimizer_updates=0,training_started=False,quality_claim=False,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('spec','archive','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);main(p.parse_args())
