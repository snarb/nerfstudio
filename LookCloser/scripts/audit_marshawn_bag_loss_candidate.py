"""Actual63-camera CPU adapter, gradient, RNG and restart proof for bag RGB weights."""
import argparse,copy,hashlib,json,os,pickle,random,sys
from pathlib import Path
from types import SimpleNamespace

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(a):
    os.environ['CUDA_VISIBLE_DEVICES']='';scripts=a.candidate/'code/LookCloser/scripts';sys.path.insert(0,str(scripts))
    import numpy as np
    import torch
    import marshawn_tuning_sampling as sampling
    import marshawn_bag_loss_correction as correction
    spec=json.loads((a.candidate/'spec.json').read_text());manifest=a.candidate/'bag_loss_source/manifest.json';inventory=json.loads((a.candidate/'source_manifest.json').read_text())['files_sha256']
    for module in [sampling,correction]:
        path=Path(module.__file__).resolve();assert path.parent==scripts.resolve()and sha(path)==inventory['LookCloser/scripts/'+path.name]
    tf=json.loads((a.data/'transforms.json').read_text());train=set(tf['train_filenames']);names=[a.data/r['file_path']for r in tf['frames']if r['file_path']in train];assert len(names)==63
    def build():
        pipe=SimpleNamespace(datamanager=SimpleNamespace(train_pixel_sampler=SimpleNamespace(sample=lambda *a,**k:None,sample_count=0,num_rays_per_batch=4096,config=SimpleNamespace(keep_full_image=False)),train_dataset=SimpleNamespace(image_filenames=names)),model=SimpleNamespace(config=SimpleNamespace(reconstruction_loss_type='charbonnier'),get_loss_dict=lambda *a,**k:{'rgb_loss':None}))
        return pipe,sampling.SamplingAdapter(pipe,a.data,'actor','body80',64)
    def rng_digest():return hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state()))+torch.get_rng_state().numpy().tobytes()).hexdigest()
    reference_pipe,reference=build();candidate_pipe,candidate=build();before=rng_digest();receipt=correction.apply_correction(candidate,manifest,spec['bag_loss_correction']['manifest_sha256']);assert before==rng_digest()
    assert reference.source_records==candidate.source_records
    metadata=json.loads(manifest.read_text());bycamera={r['camera']:r for r in metadata['records']};changed_masks=[];counts=[]
    for source,left,right in zip(candidate.source_records,reference.rows,candidate.rows):
        name=source['camera'];changed=np.zeros(left['grey'].shape,bool)
        if name in bycamera:changed=np.load(manifest.parent/bycamera[name]['exclusion'],allow_pickle=False)&left['grey']
        changed_masks.append(changed)
        for key in ['core','band','valid','black','grey','actor_union','W32','W64']:np.testing.assert_array_equal(left[key],right[key])
        for key in ['core','band','valid']:np.testing.assert_array_equal(left['pools'][key],right['pools'][key])
        np.testing.assert_array_equal(left['fas']['probabilities'],right['fas']['probabilities'])
        for ca,cb in zip(left['fas']['cells'],right['fas']['cells']):
            assert len(ca)==len(cb)
            for va,vb in zip(ca,cb):np.testing.assert_array_equal(va,vb)
        np.testing.assert_array_equal(left['weight'][~changed],right['weight'][~changed]);assert (right['weight'][changed]==0).all()
        counts.append(dict(camera=name,modified_pixels=int(changed.sum()),unchanged_confidence_outside_exact_intersection=True))
    assert sum(r['modified_pixels']for r in counts)==6522
    # Independent sparse expectation of ONLY removed original confidence.
    # This does not reuse the candidate's recomputation of full weight means.
    removed_expectations=[]
    for left,changed in zip(reference.rows,changed_masks):
        sparse=(left['weight']*changed).reshape(-1)
        components=[]
        for kind in ['core','band','valid']:
            ids=reference._pool(left,kind);components.append(float(np.sum(sparse[ids],dtype=np.float64)/len(ids)))
        fas=left['fas'];fas_removed=0.
        if fas['probabilities'].sum():
            for level,cells in enumerate(fas['cells']):
                if not cells:continue
                for ids in cells:fas_removed+=float(fas['probabilities'][level])*float(np.sum(sparse[ids],dtype=np.float64))/(len(cells)*len(ids))
        else:fas_removed=components[0]
        removed_expectations.append([components[0],fas_removed,components[1],components[2]])
    h,w=reference.rows[0]['grey'].shape
    # Broadcast dummy RGB is sufficient to exercise actual indexing/loss math;
    # real semantic rasters, proposals and frequency maps are loaded unchanged.
    batch=dict(image=torch.zeros((1,1,1,3),dtype=torch.uint8).expand(63,h,w,3),image_idx=torch.arange(63));phases=[];total_bag_draws=0
    for step in [0,4096,8192,30376]:
        reference.set_step(step);candidate.set_step(step);np.random.seed(42+step);state=np.random.get_state()
        left=reference.sample(batch);after_left=rng_digest();np.random.set_state(state);right=candidate.sample(batch);after_right=rng_digest();assert after_left==after_right
        torch.testing.assert_close(left['indices'],right['indices'],rtol=0,atol=0)
        ids=right['indices'].numpy();modified=np.array([changed_masks[c][y,x]for c,y,x in ids]);total_bag_draws+=int(modified.sum())
        np.testing.assert_array_equal(left['actor_confidence'].numpy()[~modified],right['actor_confidence'].numpy()[~modified]);assert (right['actor_confidence'].numpy()[modified]==0).all()
        norm0,norm1=left['actor_proposal_mean_weight'],right['actor_proposal_mean_weight'];assert 0<norm1<norm0<=1
        independent_delta=float(np.mean(removed_expectations,axis=0)@sampling.body80_quotas(4096,step)/4096)
        assert abs((norm0-norm1)-independent_delta)<1e-12
        p0=torch.full((4096,3),.5,requires_grad=True);p1=p0.detach().clone().requires_grad_(True)
        reference_pipe.model.get_loss_dict({'rgb':p0},left)['rgb_loss'].backward();candidate_pipe.model.get_loss_dict({'rgb':p1},right)['rgb_loss'].backward()
        assert bool(torch.isfinite(p1.grad).all())and torch.equal(p1.grad[modified],torch.zeros_like(p1.grad[modified]))
        torch.testing.assert_close(p1.grad[~modified],p0.grad[~modified]*(norm0/norm1),rtol=2e-6,atol=1e-10)
        phases.append(dict(step=step,actual_rays=4096,indices_identical=True,RNG_after_draw_identical=True,qualified_bag_rays=int(modified.sum()),reference_normalizer=norm0,corrected_normalizer=norm1,independent_sparse_removed_expectation=independent_delta,normalizer_delta_error=abs((norm0-norm1)-independent_delta),zero_bag_RGB_gradient=True,other_gradients_only_exact_normalization_scale=True))
    assert total_bag_draws>0
    saved=copy.deepcopy(candidate.state_dict());np_state=np.random.get_state();expected=candidate.sample(batch);expected_state=copy.deepcopy(candidate.state_dict())
    del reference,reference_pipe
    fresh_pipe,fresh=build();correction.apply_correction(fresh,manifest,spec['bag_loss_correction']['manifest_sha256']);fresh.load_state_dict(saved);np.random.set_state(np_state);actual=fresh.sample(batch)
    for key in ['indices','actor_confidence']:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
    assert actual['actor_proposal_mean_weight']==expected['actor_proposal_mean_weight']and fresh.state_dict()==expected_state
    bad=copy.deepcopy(saved);bad['bag_loss_correction']['manifest_sha256']='changed'
    try:fresh.load_state_dict(bad)
    except ValueError:pass
    else:raise AssertionError('Tampered restart accepted')
    assert not torch.cuda.is_initialized()
    a.output.parent.mkdir(exist_ok=True,parents=True)
    result=dict(passed=True,CPU_only=True,actual_frozen_63_camera_SamplingAdapter=True,source_manifest_sha256=sha(a.candidate/'source_manifest.json'),spec_sha256=sha(a.candidate/'spec.json'),correction_manifest_sha256=sha(manifest),script_sha256=sha(__file__),source_sampling_sha256=sha(sampling.__file__),source_correction_sha256=sha(correction.__file__),changes=counts,application=receipt,phases=phases,installation_RNG_unchanged=True,fresh_adapter_resume_exact=True,tampered_resume_rejected=True,unmodified_model_and_fixed_admission=True,synthetic_RGB_for_CPU_gradient_only=True,GPU_model_optimizer_created=False,limitations=['This verifies source/runtime algebra, not rendered quality or actual Adam updates for the new objective.','Original proposal labels intentionally still sample the removed falsegrey pixels; only effective RGB confidence changes.','Unknown shells and unsegmented hair remain unchanged; geometric quality is unproven.'])
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,modified_pixels=6522,phases=phases,output=str(a.output))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
