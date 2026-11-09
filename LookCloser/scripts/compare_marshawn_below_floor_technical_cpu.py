"""Strict restore/point-stream checks; report CUDA trajectory drift without promotion."""
import argparse
import json
import os
from pathlib import Path
import sys


def main(root):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    sc=root.parent
    sys.path[:0]=[str(sc/'distortion03_warm_continuation_candidate_v1/code/LookCloser/scripts')]
    import torch
    from train_marshawn_semantic_late_distortion import tree_digest,adam_steps
    read=lambda p:json.loads(Path(p).read_text())
    anchor=read(sc/'distortion03_warm_continuation_candidate_v1/anchor_audit.json')
    keys=['pipeline','optimizers','schedulers','scalers','rng_state','tuning_runtime','actual_occupancy_capture']
    cases=[arm+'_'+replica for arm in ['control0','prior1e4','prior1e3']for replica in ['resume64','full64']]
    records={};payloads={}
    for case in cases:
        out=root/'runs'/case;complete=read(out/'complete.json');assert complete['passed']and complete['attempts']==64
        source=read(out/'resume_proof_126316.json')
        assert all(source[k]==anchor['fullstate_hashes'][k]for k in keys)
        payload=torch.load(out/'checkpoints/step-000126380.ckpt',map_location='cpu',weights_only=False)
        runtime=payload['below_floor_density_fork_runtime'];checks=runtime['prior_checks']
        assert len(checks)==64 and runtime['below_floor_prior_state']['draws']==64
        assert all(q['global_RNG_unchanged_by_prior']and q['finite_auxiliary']and q['native_point_queries']==1024 for q in checks)
        probe=[q for q in checks if 'actual_prior_parameter_gradients_finite'in q]
        assert probe and all(q['actual_prior_parameter_gradients_finite']for q in probe)
        assert all(q['source_W64_exact']and q['actual_distortion']==.1 and q['actual_FR']==.3 for q in complete['source_control_checks'])
        if case.startswith('control0_'):assert all(q['actual_prior_parameter_gradient_max']==0. for q in probe)
        records[case]=dict(source_seven_states_exact=True,actual_Adam=adam_steps(payload['optimizers']),
            coefficient=runtime['below_floor_coefficient'],prior_packets=[q['packet']for q in checks],
            private_state_sha256=tree_digest(runtime['below_floor_prior_state']),
            private_RNG_sha256=tree_digest(runtime['below_floor_prior_state']['generator_state']),
            actual_prior_gradient_probes=probe,
            sampling_packets=runtime['sampling_segment']['packets'],
            final_global_RNG_sha256=tree_digest(payload['rng_state']),
            final_sampler_sha256=tree_digest(payload['tuning_runtime']['sampling_state']))
        payloads[case]=payload
    resume_out=root/'runs/prior1e3_resume64'
    first=torch.load(resume_out/'checkpoints/step-000126348.ckpt',map_location='cpu',weights_only=False)
    resume_proof=read(resume_out/'resume_proof_126348.json')
    assert all(resume_proof[k]==tree_digest(first[k])for k in keys)
    first_runtime=first['below_floor_density_fork_runtime']
    assert resume_proof['confidence_state_sha256']==tree_digest(first_runtime['confidence_state'])
    assert resume_proof['model_cache_sha256']==tree_digest(first_runtime['model_cache'])
    middle_Adam=adam_steps(first['optimizers']);final_Adam=adam_steps(payloads['prior1e3_resume64']['optimizers'])
    source_Adam=read(resume_out/'resume_proof_126316.json')['Adam']
    assert all(min(source_Adam[k])<min(middle_Adam[k])<min(final_Adam[k])for k in middle_Adam)
    private_resume=read(resume_out/'private_prior_resume_proof_step-000126348.json')
    assert private_resume['private_state_sha256']==tree_digest(first['below_floor_density_fork_runtime']['below_floor_prior_state'])
    assert private_resume['draws']==32
    first_packets=first['below_floor_density_fork_runtime']['sampling_segment']['packets']
    resumed_packets=first_packets+records['prior1e3_resume64']['sampling_packets']
    full_packets=records['prior1e3_full64']['sampling_packets']
    private_packets=[r['prior_packets']for r in records.values()]
    assert all(v==private_packets[0]for v in private_packets[1:])
    assert len(resumed_packets)==len(full_packets)==64
    for r in records.values():r.pop('prior_packets')
    def numeric(a,b):
        report=[]
        def walk(x,y,path):
            if torch.is_tensor(x):
                assert torch.is_tensor(y)and x.shape==y.shape and x.dtype==y.dtype
                if x.is_floating_point():
                    d=x.double()-y.double();report.append(dict(path=path,count=x.numel(),max_abs=float(d.abs().max()) if d.numel() else 0.,sum_squared=float(d.square().sum())))
                elif not torch.equal(x,y):report.append(dict(path=path,nonfloat_mismatches=int((x!=y).sum())))
            elif isinstance(x,dict):
                assert x.keys()==y.keys()
                for k in x:walk(x[k],y[k],path+'/'+str(k))
            elif isinstance(x,(list,tuple)):
                assert type(x)==type(y)and len(x)==len(y)
                for i,(u,v)in enumerate(zip(x,y)):walk(u,v,path+'/'+str(i))
            elif isinstance(x,(int,float)):
                if x!=y:report.append(dict(path=path,scalar_abs=abs(x-y)))
            else:assert x==y
        walk(a,b,'')
        floats=[q for q in report if'count'in q];count=sum(q['count']for q in floats)
        return dict(max_abs=max([q['max_abs']for q in floats]or[0.]),RMSE=(sum(q['sum_squared']for q in floats)/max(count,1))**.5,
            nonfloat_mismatches=sum(q.get('nonfloat_mismatches',0)for q in report),details=report)
    a=payloads['prior1e3_resume64'];b=payloads['prior1e3_full64']
    arm_reports={}
    for arm in ['control0','prior1e4','prior1e3']:
        location=root/'runs'/(arm+'_resume64')
        middle=torch.load(location/'checkpoints/step-000126348.ckpt',map_location='cpu',weights_only=False)
        start=read(location/'resume_proof_126348.json')
        assert all(start[k]==tree_digest(middle[k])for k in keys)
        private_start=read(location/'private_prior_resume_proof_step-000126348.json')
        assert private_start['private_state_sha256']==tree_digest(middle['below_floor_density_fork_runtime']['below_floor_prior_state'])
        assert private_start['draws']==32
        resumed=payloads[arm+'_resume64'];independent=payloads[arm+'_full64']
        assert tree_digest(resumed['below_floor_density_fork_runtime']['below_floor_prior_state'])==tree_digest(independent['below_floor_density_fork_runtime']['below_floor_prior_state'])
        combined_packets=middle['below_floor_density_fork_runtime']['sampling_segment']['packets']+records[arm+'_resume64']['sampling_packets']
        arm_reports[arm]=dict(strict_seven_state_resume_start=True,strict_private_resume_start=True,
            private_final_state_exact=True,native_ray_history_equal=combined_packets==records[arm+'_full64']['sampling_packets'],
            final_global_RNG_equal=tree_digest(resumed['rng_state'])==tree_digest(independent['rng_state']),
            final_sampler_equal=tree_digest(resumed['tuning_runtime']['sampling_state'])==tree_digest(independent['tuning_runtime']['sampling_state']),
            pipeline_numeric_drift=numeric(resumed['pipeline'],independent['pipeline']),
            Adam_numeric_drift=numeric(resumed['optimizers'],independent['optimizers']))
        del middle
    zero=read(root/'runs/zero_source/complete.json');assert zero['passed']and zero['render_state_unchanged']
    output=dict(passed=True,CPU_only=True,CUDA_started=False,source_zero_RGB_alpha_parity=zero['render_parity'],
        strict_resume_start_seven_states_exact=True,strict_private_resume_start_exact=True,
        all_coefficients_same64_private_point_packets=True,
        resumed_vs_independent_native_ray_packets_equal=resumed_packets==full_packets,
        control_vs_candidate_native_ray_packets_equal=records['control0_full64']['sampling_packets']==full_packets,
        resumed_vs_independent_final_global_RNG_equal=tree_digest(a['rng_state'])==tree_digest(b['rng_state']),
        resumed_vs_independent_final_sampler_equal=tree_digest(a['tuning_runtime']['sampling_state'])==tree_digest(b['tuning_runtime']['sampling_state']),
        pipeline_numeric_drift=numeric(a['pipeline'],b['pipeline']),Adam_numeric_drift=numeric(a['optimizers'],b['optimizers']),
        per_arm_resume_reports=arm_reports,records=records,main_global_streams_observed_not_forced_equal=True,
        CUDA_atomic_trajectory_bitwise_not_claimed=True,numeric_trajectory_requires_root_review=True,
        quality_promotion=False,production_authorized=False)
    target=root/'technical_CPU_comparison';target.mkdir(exist_ok=True)
    (target/'comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(passed=True,strict_resume_start_exact=True,private_packets_exact=True,
        pipeline_max_abs=output['pipeline_numeric_drift']['max_abs'],root_numeric_review_required=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
