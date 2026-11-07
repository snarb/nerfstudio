"""Real CPU-only source config, 63-camera sampler and strict policy roundtrip."""
import argparse,copy,dataclasses,hashlib,json,os,pickle,random,sys
from pathlib import Path
from types import SimpleNamespace

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def normalized(x):
    if isinstance(x,Path):return str(x)
    if isinstance(x,type)or callable(x):return x.__module__+'.'+x.__qualname__
    if dataclasses.is_dataclass(x):return {f.name:normalized(getattr(x,f.name))for f in dataclasses.fields(x)}
    if isinstance(x,dict):return {str(k):normalized(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [normalized(v)for v in x]
    return x

def differences(a,b,prefix=''):
    if isinstance(a,dict)and isinstance(b,dict):
        out={}
        for k in set(a)|set(b):out.update(differences(a.get(k),b.get(k),prefix+'.'+k))
        return out
    return {}if a==b else{prefix:dict(before=a,after=b)}


def main(root):
    os.environ['CUDA_VISIBLE_DEVICES']='';sys.path[:0]=[str(root/'code/LookCloser/scripts'),str(root/'code')]
    import numpy as np,torch,yaml
    import marshawn_tuning_sampling as sampling
    from marshawn_margin_continuation_policy import ARMS,MarginPolicy
    from marshawn_margin_continuation_factory import build_config
    from train_marshawn_margin_continuation import validate_spec
    from marshawn_tuning_metrics import PlateauTracker
    torch.set_num_threads(4)
    manifest=root/'margin/manifest.json';digest=sha(manifest);parent=json.loads((root/'anchor/spec.json').read_text());anchor=json.loads((root/'anchor_audit.json').read_text());data=Path(parent['data']);tf=json.loads((data/'transforms.json').read_text());train=set(tf['train_filenames']);names=[data/f['file_path']for f in tf['frames']if f['file_path']in train]
    ref=build_config(parent,anchor,root/'anchor/checkpoint.ckpt',root/'reference_output');refnorm=normalized(ref);(root/'configurations').mkdir(exist_ok=True);(root/'configurations/reference_resume.yml').write_text(yaml.dump(ref));cfgdiff={}
    for phase in ('zero','technical','production'):
        for arm in ARMS:
            spec=json.loads((root/'specs'/f'{phase}_{arm}/spec.json').read_text());validate_spec(spec);cfg=build_config(parent,anchor,root/'anchor/checkpoint.ckpt',Path(spec['output']));d=differences(refnorm,normalized(cfg));assert set(d)=={'.output_dir'},d;cfgdiff[phase+'_'+arm]=d;(root/'configurations'/(phase+'_'+arm+'.yml')).write_text(yaml.dump(cfg))
    archived=yaml.load((root/'anchor/config.yml').read_text(),Loader=yaml.UnsafeLoader);archive_diff=differences(normalized(archived),refnorm)
    class Field:
        _semantic_admission_installed=True
        def _normalize_positions(self,p):return p,torch.ones(p.shape[:-1],dtype=torch.bool),p.shape[:-1],p
    def build():
        pipe=SimpleNamespace(datamanager=SimpleNamespace(train_pixel_sampler=SimpleNamespace(sample=lambda *a,**k:None,sample_count=0,num_rays_per_batch=4096,config=SimpleNamespace(keep_full_image=False)),train_dataset=SimpleNamespace(image_filenames=names)),model=SimpleNamespace(config=SimpleNamespace(reconstruction_loss_type='charbonnier'),get_loss_dict=lambda *a,**k:{'rgb_loss':None}))
        return sampling.SamplingAdapter(pipe,data,'actor','body80',64)
    def rng():return hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state()))+torch.get_rng_state().numpy().tobytes()).hexdigest()
    def structures(adapter):
        h=hashlib.sha256()
        for row in adapter.rows:
            for key,value in sorted(row.items()):
                if isinstance(value,np.ndarray):h.update(key.encode());h.update(str(value.shape).encode());h.update(value.tobytes())
        return h.hexdigest()
    adapter=build();baseline=structures(adapter);initial=copy.deepcopy(adapter.state_dict());h,w=adapter.rows[0]['core'].shape;batch=dict(image=torch.zeros((1,1,1,3),dtype=torch.uint8).expand(63,h,w,3),image_idx=torch.arange(63));reference=None;policies={}
    for arm in ARMS:
        adapter.load_state_dict(initial);adapter.set_step(126317);before=rng();model=SimpleNamespace(field=Field(),device=torch.device('cpu'));policy=MarginPolicy(model,adapter,manifest,digest,arm);assert rng()==before and structures(adapter)==baseline
        np.random.seed(71242);sample=adapter.sample(batch);after=rng();indices=sample['indices'].clone()
        if reference is None:reference=(indices,after,sample['actor_confidence'].clone())
        else:assert torch.equal(indices,reference[0])and after==reference[1]and torch.equal(sample['actor_confidence'],reference[2])
        normalizer=policy.normalizer(126317,4096);assert abs(sample['actor_proposal_mean_weight']-normalizer)<1e-12
        state=copy.deepcopy(adapter.state_dict());gate=policy.state_dict();numpy_state=np.random.get_state();expected=adapter.sample(batch);expected_state=copy.deepcopy(adapter.state_dict());fresh=build();fresh.load_state_dict(state);freshmodel=SimpleNamespace(field=Field(),device=torch.device('cpu'));restored=MarginPolicy(freshmodel,fresh,manifest,digest,arm);restored.load_state_dict(gate);np.random.set_state(numpy_state);replayed=fresh.sample(batch)
        for key in ('indices','actor_confidence'):assert torch.equal(expected[key],replayed[key])
        assert expected['actor_proposal_mean_weight']==replayed['actor_proposal_mean_weight']and fresh.state_dict()==expected_state
        for key,value in [('binding',dict(gate['binding'],arm='wrong')),('last_normalizer',gate['last_normalizer']+.001)]:
            bad=copy.deepcopy(gate);bad[key]=value
            try:policy.load_state_dict(bad)
            except ValueError:pass
            else:raise AssertionError('Tampered selector/normalizer accepted')
        policies[arm]=dict(normalizer=normalizer,original_W64_pools_FAS_exact=True,same_first_indices_weights_and_RNG=True,installation_RNG_unchanged=True,fresh_private_sampler_and_selector_roundtrip_exact=True,tampered_selector_and_normalizer_rejected=True)
        del fresh,restored
    tracker=PlateauTracker(anchor_step=130316)
    for step in (130316,135316,140316,145316,150316):tracker.observe(step,.1,.02)
    assert tracker.stopped and tracker.last_step==150316 and not torch.cuda.is_initialized()
    result=dict(passed=True,CPU_only=True,manifest_sha256=digest,source_sampling_sha256=sha(sampling.__file__),source_structure_sha256=baseline,config_differences_from_common_actual_resume=cfgdiff,actual_resume_vs_historical_initial_config=archive_diff,actual_common_distortion=.1,restored_FR_expected=.3,source_Adam=anchor['actual_Adam'],policies=policies,separate_tracker=tracker.to_dict(),minimum_no_improvement_attempts=24000,zero_FHD_fullmodel_and_density_gates='PENDING_NOT_RUN',technical_actual_Adam32_plus32='PENDING_NOT_RUN',GPU_or_training_started=False)
    (root/'CPU_readiness.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,normalizers={k:v['normalizer']for k,v in policies.items()},config_changes=sorted(archive_diff))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
