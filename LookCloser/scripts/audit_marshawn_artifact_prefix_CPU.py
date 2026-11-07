"""CPU config, real63-camera loss-policy and checkpoint-binding feasibility."""
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
    from marshawn_artifact_prefix_rgb import ConfidencePolicy,structure_sha
    from marshawn_artifact_prefix_tau import ARMS,QualifiedPrefixTau,validate_targets
    from marshawn_artifact_prefix_factory import build_config
    from train_marshawn_artifact_prefix import validate_spec
    from marshawn_tuning_metrics import PlateauTracker
    import train_marshawn_semantic_admission as base
    manifest=root/'confidence/manifest.json';digest=sha(manifest);meta=json.loads(manifest.read_text());parent=json.loads((root/'anchor/spec.json').read_text());anchor=json.loads((root/'anchor_audit.json').read_text());data=Path(parent['data']);tf=json.loads((data/'transforms.json').read_text());train=set(tf['train_filenames']);names=[data/f['file_path']for f in tf['frames']if f['file_path']in train]
    ref=build_config(parent,anchor,root/'anchor/checkpoint.ckpt',root/'reference_output');refnorm=normalized(ref);(root/'configurations').mkdir(exist_ok=True);(root/'configurations/reference_resume.yml').write_text(yaml.dump(ref))
    cfgdiff={}
    for phase in ('zero','technical','production'):
        for arm in ARMS:
            spec=json.loads((root/'specs'/f'{phase}_{arm}/spec.json').read_text());validate_spec(spec);cfg=build_config(parent,anchor,root/'anchor/checkpoint.ckpt',Path(spec['output']));d=differences(refnorm,normalized(cfg));assert set(d)=={'.output_dir'},d;cfgdiff[phase+'_'+arm]=d;(root/'configurations'/(phase+'_'+arm+'.yml')).write_text(yaml.dump(cfg))
    archived=yaml.load((root/'anchor/config.yml').read_text(),Loader=yaml.UnsafeLoader);archive_diff=differences(normalized(archived),refnorm)
    def build():
        pipe=SimpleNamespace(datamanager=SimpleNamespace(train_pixel_sampler=SimpleNamespace(sample=lambda *a,**k:None,sample_count=0,num_rays_per_batch=4096,config=SimpleNamespace(keep_full_image=False)),train_dataset=SimpleNamespace(image_filenames=names)),model=SimpleNamespace(config=SimpleNamespace(reconstruction_loss_type='charbonnier'),get_loss_dict=lambda *a,**k:{'rgb_loss':None}))
        return sampling.SamplingAdapter(pipe,data,'actor','body80',64)
    def rng():return hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state()))+torch.get_rng_state().numpy().tobytes()).hexdigest()
    adapter=build();baseline=structure_sha(adapter);initial=copy.deepcopy(adapter.state_dict());h,w=adapter.rows[0]['core'].shape;batch=dict(image=torch.zeros((1,1,1,3),dtype=torch.uint8).expand(63,h,w,3),image_idx=torch.arange(63));reference=None;policies={}
    for arm in ARMS:
        adapter.load_state_dict(initial);adapter.set_step(126317);before=rng();policy=ConfidencePolicy(adapter,manifest,digest,'original');assert rng()==before and structure_sha(adapter)==baseline
        np.random.seed(71242);sample=adapter.sample(batch);after=rng();indices=sample['indices'].clone()
        if reference is None:reference=(indices,after)
        else:assert torch.equal(indices,reference[0])and after==reference[1]
        normalizer=policy.normalizer(126317,4096);assert abs(sample['actor_proposal_mean_weight']-normalizer)<1e-12
        saved_sampling=copy.deepcopy(adapter.state_dict());saved_confidence=policy.state_dict();saved_np=np.random.get_state();expected=adapter.sample(batch);expected_state=copy.deepcopy(adapter.state_dict())
        fresh=build();fresh.load_state_dict(saved_sampling);fresh_policy=ConfidencePolicy(fresh,manifest,digest,'original');fresh_policy.load_state_dict(saved_confidence);np.random.set_state(saved_np);replayed=fresh.sample(batch)
        for key in ('indices','actor_confidence'):assert torch.equal(expected[key],replayed[key])
        assert expected['actor_proposal_mean_weight']==replayed['actor_proposal_mean_weight']and fresh.state_dict()==expected_state
        bad=copy.deepcopy(saved_confidence);bad['binding']['manifest_sha256']='changed'
        try:policy.load_state_dict(bad)
        except ValueError:pass
        else:raise AssertionError('Tampered confidence source accepted')
        bad=copy.deepcopy(saved_confidence);bad['last_normalizer']+=.0001
        try:policy.load_state_dict(bad)
        except ValueError:pass
        else:raise AssertionError('Tampered normalizer accepted')
        policies[arm]=dict(normalizer=normalizer,confidence_state_sha256=hashlib.sha256(json.dumps(saved_confidence,sort_keys=True).encode()).hexdigest(),same_packet_and_RNG=True,installation_RNG_unchanged=True,fresh_adapter_profile_and_normalizer_roundtrip_exact=True,tampered_source_and_normalizer_rejected=True)
        del fresh,fresh_policy
    targets=validate_targets(root/'root_selected196_rays.json',data,Path(parent['semantic_admission']['manifest']).with_name('support_manifest.json'))
    class ToyField:
        def __init__(self):self.scale=torch.tensor(1.,requires_grad=True);self.queries=0
        def get_density(self,rays):
            self.queries+=1;return self.scale.expand(len(rays.frustums.origins),1),None
    private={};gradients={}
    global_before=torch.get_rng_state().clone()
    for arm,coefficient in ARMS.items():
        field=ToyField();model=SimpleNamespace(device=torch.device('cpu'),field=field)
        aux=QualifiedPrefixTau(model,targets,coefficient);value=aux.loss()
        if coefficient==0.:assert field.queries==0 and value.item()==0
        else:
            value.backward();gradients[arm]=field.scale.grad.item();assert np.isfinite(gradients[arm])and gradients[arm]>0 and field.queries==1
        for _ in range(31):aux.draw()
        state=copy.deepcopy(aux.state_dict());expected=[aux.draw()[2]for _ in range(32)]
        fresh=QualifiedPrefixTau(model,targets,coefficient);fresh.load_state_dict(state)
        assert expected==[fresh.draw()[2]for _ in range(32)]
        assert torch.equal(aux.generator.get_state(),fresh.generator.get_state())and aux.chain==fresh.chain and aux.ids_chain==fresh.ids_chain and aux.jitter_chain==fresh.jitter_chain
        private[arm]=dict(draws=fresh.draws,chain=fresh.chain,ids_chain=fresh.ids_chain,jitter_chain=fresh.jitter_chain,generator=hashlib.sha256(fresh.generator.get_state().numpy().tobytes()).hexdigest(),control_density_query_skipped=field.queries==0 if coefficient==0 else None)
    assert torch.equal(global_before,torch.get_rng_state())
    for key in ('draws','chain','ids_chain','jitter_chain','generator'):assert len({r[key]for r in private.values()})==1
    assert abs(gradients['tau01']/gradients['tau001']-10)<1e-5
    (root/'CPU_private_integral.json').write_text(json.dumps(dict(passed=True,private=private,gradient_ratio=gradients['tau01']/gradients['tau001'],sealed_native_tau_method_reused=True,global_CPU_RNG_unchanged=True,no_GPU=True),indent=2)+'\n')
    tracker=PlateauTracker(anchor_step=130316)
    for step in (130316,135316,140316,145316,150316):tracker.observe(step,.1,.02)
    assert tracker.stopped and tracker.last_step==150316 and tracker.last_step-126316==24000
    assert structure_sha(adapter)==baseline and not torch.cuda.is_initialized()
    result=dict(passed=True,CPU_only=True,manifest_sha256=digest,source_sampling_sha256=sha(sampling.__file__),source_structure_sha256=baseline,config_differences_from_common_actual_resume=cfgdiff,actual_resume_vs_historical_initial_config=archive_diff,actual_common_distortion=.1,restored_FR_expected=.3,policies=policies,separate_tracker=tracker.to_dict(),minimum_no_improvement_attempts=24000,zero_FHD_and_full_model_loader_gates='PENDING_NOT_RUN',technical_actual_Adam_32_plus32='PENDING_NOT_RUN',root_source196_qualification='PASS_CONDITIONAL_SOFT_PREFIX_ONLY',GPU_or_training_started=False)
    (root/'CPU_readiness.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,normalizers={k:v['normalizer']for k,v in policies.items()},archive_config_changes=sorted(archive_diff))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
