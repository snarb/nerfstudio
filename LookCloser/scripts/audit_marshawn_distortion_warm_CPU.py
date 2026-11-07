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
    from marshawn_distortion_warm_rgb import ConfidencePolicy,structure_sha
    from marshawn_distortion_warm_factory import ARMS,build_effective_config
    from marshawn_distortion_warm_factory import build_config
    from train_marshawn_distortion_warm import validate_spec
    from marshawn_tuning_metrics import PlateauTracker
    import train_marshawn_semantic_admission as base
    manifest=root/'confidence/manifest.json';digest=sha(manifest);meta=json.loads(manifest.read_text());parent=json.loads((root/'anchor/spec.json').read_text());anchor=json.loads((root/'anchor_audit.json').read_text());data=Path(parent['data']);tf=json.loads((data/'transforms.json').read_text());train=set(tf['train_filenames']);names=[data/f['file_path']for f in tf['frames']if f['file_path']in train]
    ref=build_config(parent,anchor,root/'anchor/checkpoint.ckpt',root/'reference_output');refnorm=normalized(ref);(root/'configurations').mkdir(exist_ok=True);(root/'configurations/reference_resume.yml').write_text(yaml.dump(ref))
    cfgdiff={}
    for phase in ('zero','technical','production'):
        for arm in ARMS:
            spec=json.loads((root/'specs'/f'{phase}_{arm}/spec.json').read_text());validate_spec(spec);cfg=build_effective_config(parent,anchor,root/'anchor/checkpoint.ckpt',Path(spec['output']),arm);d=differences(refnorm,normalized(cfg));expected={'.output_dir'}if ARMS[arm]==.1 else{'.output_dir','.pipeline.model.distortion_loss_mult'};assert set(d)==expected,d;cfgdiff[phase+'_'+arm]=d;(root/'configurations'/(phase+'_'+arm+'.yml')).write_text(yaml.dump(cfg))
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
    from nerfstudio.models.lookcloser import LookCloserModel
    w=torch.tensor([[.1],[.7],[.1],[.000001],[.000002]],dtype=torch.float32,requires_grad=True)
    starts=torch.tensor([[0.],[.2],[.4],[.1],[.3]],dtype=torch.float32);ends=starts+.1;ids=torch.tensor([0,0,0,1,1],dtype=torch.int64)
    import nerfacc
    native_pack_info=nerfacc.pack_info
    def cpu_pack_info(indices,num_rays):
        if indices.is_cuda:raise ValueError('CPU metadata adapter is test-only')
        counts=torch.bincount(indices,minlength=num_rays)
        return torch.stack((torch.cumsum(counts,0)-counts,counts),dim=-1)
    # Installed nerfacc supports CUDA only. Exercise the unmodified model's
    # FP64 arithmetic with exact CPU CSR metadata, then restore the native API.
    nerfacc.pack_info=cpu_pack_info
    try:raw=LookCloserModel._packed_distortion_loss(starts,ends,w,ids,2).mean()
    finally:nerfacc.pack_info=native_pack_info
    assert nerfacc.pack_info is native_pack_info

    source=.1*raw;treatment=.3*raw
    g1=torch.autograd.grad(source,w,retain_graph=True)[0];g3=torch.autograd.grad(treatment,w,retain_graph=True)[0]
    torch.testing.assert_close(treatment,3*source,rtol=1e-6,atol=1e-10);torch.testing.assert_close(g3,3*g1,rtol=1e-6,atol=1e-9)
    assert torch.isfinite(g3).all()and raw>=0
    (root/'CPU_packed_factor.json').write_text(json.dumps(dict(passed=True,native_original_packed_function_used=True,CPU_pack_info_metadata_adapter_used=True,native_CUDA_pack_info_not_tested_on_CPU=True,pack_info_adapter_restored_before_exit=True,source_weight_dtype='float32',native_FP64_prefix_arithmetic_unchanged=True,source_coefficient=.1,treatment_coefficient=.3,expected_factor=3,gradient_max_error=float((g3-3*g1).abs().max()),CUDA_initialized=torch.cuda.is_initialized(),no_sigma_clamp_or_precision_override=True),indent=2)+'\n')
    tracker=PlateauTracker(anchor_step=130316)
    for step in (130316,135316,140316,145316,150316):tracker.observe(step,.1,.02)
    assert tracker.stopped and tracker.last_step==150316 and tracker.last_step-126316==24000
    assert structure_sha(adapter)==baseline and not torch.cuda.is_initialized()
    result=dict(passed=True,CPU_only=True,manifest_sha256=digest,source_sampling_sha256=sha(sampling.__file__),source_structure_sha256=baseline,config_differences_from_common_actual_resume=cfgdiff,actual_resume_vs_historical_initial_config=archive_diff,actual_common_distortion=.1,restored_FR_expected=.3,policies=policies,separate_tracker=tracker.to_dict(),minimum_no_improvement_attempts=24000,zero_FHD_and_full_model_loader_gates='PENDING_NOT_RUN',technical_actual_Adam_32_plus32='PENDING_NOT_RUN',new_labels_or_RGB_changes=False,effective_distortion_factors=ARMS,GPU_or_training_started=False)
    (root/'CPU_readiness.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,normalizers={k:v['normalizer']for k,v in policies.items()},archive_config_changes=sorted(archive_diff))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
