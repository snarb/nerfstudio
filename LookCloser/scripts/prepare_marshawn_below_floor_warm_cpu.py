"""Derive a narrow technical warm-RGB fork; original1495/qualified framework immutable."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys

C=Path('/fsx/tmp/marshawn_001660_hd/tuning_20261106')
SC='/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106'
OLD=C/'distortion03_warm_continuation_candidate_v1'
P=C/'below_floor_density_warm_technical_plan_v1';R=SC+'/'+P.name
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2)+'\n')


def derive_worker():
    original=(OLD/'code/LookCloser/scripts/train_marshawn_distortion_warm.py').read_text();text=original
    def replace(old,new):
        nonlocal text
        if text.count(old)!=1:raise ValueError('Exact qualified source insertion missing:'+old[:100])
        text=text.replace(old,new)
    replace('"""Matched full-state .1/.3 distortion forks; original scientific loop and plateau criterion."""',
            '"""Opt-in below-floor density prior; original native warm-RGB loop and callbacks."""')
    replace("'density_selector'}", "'density_selector','below_floor_density_prior'}")
    replace("    return s\n", """    p=s['below_floor_density_prior']
    if s['arm']!='control01' or set(p)!={'coefficient','table','table_sha256','seed','count','focused_probability','dt','band','source_grid_sha256'}:raise ValueError('Original .1 distortion and exact below-floor prior required')
    if p['coefficient']not in (0.,1e-4,1e-3)or(p['seed'],p['count'],p['focused_probability'],p['dt'])!=(42,1024,.8,.0015625):raise ValueError('Declared fixed prior only')
    if p['band']!=[[-.84,-.21,-.07],[.15,.42,-.05]]or sha(p['table'])!=p['table_sha256']:raise ValueError('Frozen below-.05 table differs')
    if p['source_grid_sha256']!='7db15fdda89f0661e767e53dd97ab34f3346550aa319d4363bdd6332b3cbac0f':raise ValueError('Original actual active-grid SHA differs')
    return s
""")
    replace('    from marshawn_distortion_warm_rgb import ConfidencePolicy',
            '    from marshawn_distortion_warm_rgb import ConfidencePolicy\n    from marshawn_below_floor_density_prior import FrozenBelowFloorPrior')
    replace('require_controller_lease):','require_controller_lease,FrozenBelowFloorPrior):')
    replace("loaded.get('distortion_fork_runtime')","loaded.get('below_floor_density_fork_runtime')")
    replace("    target=SOURCE_STEP+64 if s['mode']=='technical'else None;", """    prior_spec=s['below_floor_density_prior']
    prior=FrozenBelowFloorPrior(prior_spec['table'],seed=prior_spec['seed'],count=prior_spec['count'],focused_probability=prior_spec['focused_probability'],dt=prior_spec['dt'])
    coefficient=prior_spec['coefficient']
    if child:
        if lineage['below_floor_coefficient']!=coefficient:raise ValueError('Child prior coefficient differs')
        prior.load_state_dict(lineage['below_floor_prior_state'])
        if tree_digest(prior.state_dict())!=tree_digest(lineage['below_floor_prior_state']):raise ValueError('Private prior resume-start state differs')
    atomic(out/f"private_prior_resume_proof_{checkpoint.stem}.json",dict(coefficient=coefficient,private_state_sha256=tree_digest(prior.state_dict()),draws=prior.draws,source_global_RNG_sha256=tree_digest(_capture_rng_state()),strict_restore_start=True))
    prior_checks=copy.deepcopy(lineage['prior_checks'])if child else[]
    original_loss_with_controls=m.get_loss_dict
    checked_prior=[False]
    def loss_with_density_prior(outputs,batch,metrics_dict=None):
        losses=original_loss_with_controls(outputs,batch,metrics_dict)
        if not m.training:return losses
        if prior.draws!=sampler.step-SOURCE_STEP-1:raise ValueError('Exactly one private point packet per native train iteration required')
        global_before=tree_digest(_capture_rng_state())
        xyz,packet=prior.draw(m.device)
        sigma=m.field.density_fn(xyz).reshape(-1)
        weighted,auxiliary=prior.objective(sigma,coefficient)
        check=dict(step=sampler.step,coefficient=coefficient,packet=packet,native_point_queries=len(xyz),
            actual_sigma_min=float(sigma.detach().min()),actual_sigma_max=float(sigma.detach().max()),
            finite_auxiliary=bool(torch.isfinite(auxiliary)),RGB_loss_tensor_preserved=True,
            private_RNG_state_sha256=tree_digest(prior.generator.get_state()),source_sampler_unchanged=True)
        if s['mode']=='technical'and not checked_prior[0]:
            parameters=[v for v in m.parameters()if v.requires_grad]
            probe_scale=float(trainer.grad_scaler.get_scale())
            auxiliary_gradients=torch.autograd.grad(weighted*probe_scale,parameters,retain_graph=True,allow_unused=True)
            finite=all(bool(torch.isfinite(g).all())for g in auxiliary_gradients if g is not None)
            maximum=max([float(g.detach().abs().max())/probe_scale for g in auxiliary_gradients if g is not None]or[0.])
            if not finite or(coefficient==0. and maximum!=0.):raise ValueError('Finite actual prior gradient and exact lambda0 zero contribution required')
            check.update(actual_prior_parameter_gradients_finite=finite,actual_prior_parameter_gradient_max=maximum,probe_source_AMP_scale=probe_scale,
                lambda0_parameter_gradient_exact_zero=coefficient==0. and maximum==0.,
                native_RGB_gradient_checks_unchanged=bool(control_checks))
            checked_prior[0]=True
        if tree_digest(_capture_rng_state())!=global_before:raise ValueError('Private point query consumed original global RNG')
        check['global_RNG_unchanged_by_prior']=True
        prior_checks.append(check)
        losses=dict(losses);losses['below_floor_density_prior']=weighted
        return losses
    m.get_loss_dict=loss_with_density_prior
    target=SOURCE_STEP+64 if s['mode']=='technical'else None;""")
    replace('distortion_fork_runtime=dict(effective_distortion=effective_distortion,',
            'below_floor_density_fork_runtime=dict(below_floor_coefficient=coefficient,below_floor_prior_state=prior.state_dict(),prior_checks=prior_checks,effective_distortion=effective_distortion,')
    replace('extra_field_queries=0,extra_optimizer_updates=0',
            'native_ray_path_extra_field_queries=0,declared_prior_field_queries_per_iteration=1024,extra_optimizer_updates=0')
    replace('passed=True,final_step=step,attempts=delta,plateau=',
            'passed=True,below_floor_coefficient=coefficient,below_floor_prior_state_sha256=tree_digest(prior.state_dict()),prior_checks=prior_checks,final_step=step,attempts=delta,plateau=')
    # A dry run validates contracts without source loading or CUDA.
    old="a=p.parse_args();run(a.spec,a.stop_after_attempts)"
    if old not in text:
        raise ValueError('Inspect qualified CLI before adding explicit dryrun')
    replace(old,"""p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if a.dry_run:
        s=validate_spec(read(a.spec));print(json.dumps(dict(mode=s['mode'],coefficient=s['below_floor_density_prior']['coefficient'],native_ARM1024=True,private_point_budget=1024,CUDA_started=False,source_model_loaded=False)))
    else:run(a.spec,a.stop_after_attempts)""")
    return original,text


def main():
    P.mkdir(exist_ok=True);(P/'helpers').mkdir(exist_ok=True)
    original,text=derive_worker()
    worker=P/'helpers/train_marshawn_below_floor_density_warm.py';worker.write_text(text)
    # Compile is CPU-only, before sealing any executable hash.
    compile(text,str(worker),'exec')
    helper=Path(__file__).with_name('marshawn_below_floor_density_prior.py');shutil.copyfile(helper,P/'helpers'/helper.name)
    shutil.copyfile(C/'frozen_LC_semantic_physical_plan_v3/code/marshawn_tuning_controller.py',P/'helpers/marshawn_tuning_controller.py')
    compare=Path(__file__).with_name('compare_marshawn_below_floor_technical_cpu.py')
    shutil.copyfile(compare,P/'helpers'/compare.name)
    zero_verify=Path(__file__).with_name('verify_marshawn_below_floor_zero_cpu.py')
    shutil.copyfile(zero_verify,P/'helpers'/zero_verify.name)
    table=C/'below_floor_density_prior_CPU_v1/frozen_active_below_band_cells.npz';shutil.copyfile(table,P/table.name)
    write(P/'scientific_scope.json',dict(source1495_changed=False,qualified_framework_changed=False,
        sole_objective_factor='lambda*mean(log1p(native_sigma*.0015625)) on frozen80/20 focused prior below-.05',
        sampler_policy='original legacy ARM1024, native callbacks unchanged',FR=.3,distortion=.1,W64='original',
        source_RGB_targets_unchanged=True,prior_is_not_free_space_GT=True,source_grid_sampler_is_frozen_prior_only=True,
        hard_clip=False,source_AABB_changed=False,all_permissions=False))
    (P/'qualified_worker_diff.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='qualified/train_marshawn_distortion_warm.py',tofile='new/train_marshawn_below_floor_density_warm.py')))
    template=json.loads((OLD/'specs/technical_control01/spec.json').read_text())
    cases=[('zero_source',0.,'zero')]+[(name+'_'+replica,coefficient,'technical')
        for name,coefficient in [('control0',0.),('prior1e4',1e-4),('prior1e3',1e-3)]
        for replica in ['resume64','full64']]
    wanted={name for name,_,_ in cases}
    for directory in (P/'specs').glob('*')if(P/'specs').exists()else[]:
        if directory.name not in wanted:
            release=directory/'root_release.json'
            if release.exists()and any(json.loads(release.read_text()).get(k)for k in ['technical_authorized','production_authorized']):
                raise ValueError('Never remove an authorized prior generation')
            shutil.rmtree(directory)
    spec_hashes={};release_templates={};all_bound=dict(template['frozen_source_hashes'])
    helper_remote=R+'/helpers/marshawn_below_floor_density_prior.py'
    table_remote=R+'/'+table.name
    worker_remote=R+'/helpers/'+worker.name
    for name,coefficient,mode in cases:
        s=json.loads(json.dumps(template));s.update(output=R+'/runs/'+name,mode=mode,arm='control01',
            release=R+'/specs/'+name+'/root_release.json',worker_sha256=sha(worker),parent_checkpoint=R+'/anchor/checkpoint.ckpt')
        s['frozen_source_hashes'][helper_remote]=sha(P/'helpers'/helper.name)
        s['frozen_source_hashes'][worker_remote]=sha(worker)
        s['frozen_source_hashes'][table_remote]=sha(P/table.name)
        s['below_floor_density_prior']=dict(coefficient=coefficient,table=table_remote,
            table_sha256=sha(P/table.name),seed=42,count=1024,focused_probability=.8,dt=.0015625,
            band=[[-.84,-.21,-.07],[.15,.42,-.05]],source_grid_sha256='7db15fdda89f0661e767e53dd97ab34f3346550aa319d4363bdd6332b3cbac0f')
        sp=P/'specs'/name/'spec.json';write(sp,s);spec_hashes[R+'/specs/'+name+'/spec.json']=sha(sp)
        rp=P/'specs'/name/'root_review_TEMPLATE.json';write(rp,dict(reviewer='root',technical_authorized=False,production_authorized=False,spec_sha256=sha(sp)))
        release_templates[R+'/specs/'+name+'/root_release.json']=dict(template=str(rp),template_sha256=sha(rp))
        all_bound.update(s['frozen_source_hashes'])
        for key in ['parent_checkpoint','parent_spec','anchor_audit','confidence_manifest']:
            all_bound[s[key]]=s[key+'_sha256']
    oldscripts=SC+'/distortion03_warm_continuation_candidate_v1/code/LookCloser/scripts'
    env=dict(PYTHONPATH=R+'/helpers:'+SC+'/distortion03_warm_continuation_candidate_v1/code:'+oldscripts,
        PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',
        LOOKCLOSER_GPU_LEASE_RECEIPT='{job_root}/receipt.json')
    def job(name,case,gpu,deps,partial=False):
        command=[SC+'/python',worker_remote,'--spec',R+'/specs/'+case+'/spec.json']
        if partial:command+=['--stop-after-attempts','32']
        return dict(name=name,host='silver',gpu=gpu,command=command,cwd=R+'/helpers',env=dict(env),dependencies=deps,
            progress_files=[R+'/runs/'+case+'/progress.json'],metrics_file=R+'/runs/'+case+('/latest.json'if partial else'/complete.json'))
    host=dict(address='ubuntu@silver-sable',python=SC+'/python',controller=R+'/helpers/marshawn_tuning_controller.py',root=R+'/remote_queue',gpus=[5,6,7])
    zero=job('below_floor_zero_source','zero_source',5,[])
    jobs=[]
    for arm,gpu in [('control0',5),('prior1e4',6),('prior1e3',7)]:
        first=arm+'_first32';resume=arm+'_resume32';full=arm+'_full64'
        jobs.extend([job(first,arm+'_resume64',gpu,[],True),job(resume,arm+'_resume64',gpu,[first]),job(full,arm+'_full64',gpu,[resume])])
    write(P/'zero_plan.json',dict(schema_version=1,poll_seconds=15,hosts=dict(silver=host),jobs=[zero]))
    write(P/'technical_plan.json',dict(schema_version=1,poll_seconds=15,hosts=dict(silver=host),jobs=jobs))
    # Reuse qualified external guardian core; only namespace and phase case
    # enumeration change. Production is deliberately absent from phase_plans.
    guard_dir=P/'external_guard_below_floor_v1';guard_dir.mkdir(exist_ok=True)
    old_guard=OLD/'external_guard_distortion03_v1/guardian.py';guardian=old_guard.read_text().replace('external_guard_distortion03_v1','external_guard_below_floor_v1')
    old_loop="for arm in ['control01', 'distortion03']:\n        spec=root/'specs'/f'{phase}_{arm}'/'spec.json'"
    if guardian.count(old_loop)!=1:raise ValueError('Exact guardian case enumeration missing')
    guardian=guardian.replace(old_loop,"for case in policy['phase_spec_cases'][phase]:\n        spec=root/'specs'/case/'spec.json'")
    (guard_dir/'guardian.py').write_text(guardian)
    (guard_dir/'guardian.diff').write_text(''.join(difflib.unified_diff(old_guard.read_text().splitlines(True),guardian.splitlines(True),fromfile='qualified/guardian.py',tofile='new/guardian.py')))
    policy=json.loads((OLD/'external_guard_distortion03_v1/policy.json').read_text())
    policy.update(local_root=str(P),remote_root=R,local_python='/home/pavlo/.conda/envs/nerfstudio/bin/python',
        controller_path=str(P/'helpers/marshawn_tuning_controller.py'),worker_path=worker_remote,
        guardian_sha256=sha(guard_dir/'guardian.py'),approved_gpu_indices=[5,6,7],
        approved_gpu_UUIDs={'5':'GPU-216eac5a-c2f3-4bc5-d4e8-1a6f9cbd9b9a','6':'GPU-a53fcf27-02c4-c737-9269-0623ec25a6d9','7':'GPU-9c2ea572-537d-66a7-6df3-41c3d8914d39'},
        phase_plans={'zero':'zero_plan.json','technical':'technical_plan.json'},
        phase_deadlines={'zero':900,'technical':1800},root_receipts={},
        phase_spec_cases={'zero':['zero_source'],'technical':[name for name,_,mode in cases if mode=='technical']},
        prepared_bindings={})
    for f in [P/'helpers/marshawn_tuning_controller.py',P/'helpers'/compare.name,P/'helpers'/zero_verify.name,P/'zero_plan.json',P/'technical_plan.json',P/'qualified_worker_diff.patch',
              P/'scientific_scope.json',*(P/'specs').glob('*/spec.json')]:all_bound[str(f).replace(str(C),SC)]=sha(f)
    for f in [worker,P/'helpers'/helper.name,P/'helpers'/compare.name,P/'helpers'/zero_verify.name,P/table.name,P/'zero_plan.json',P/'technical_plan.json',guard_dir/'guardian.py',*(P/'specs').glob('*/spec.json')]:
        policy['prepared_bindings'][str(f.relative_to(P))]=sha(f)
    write(guard_dir/'policy.json',policy)
    for phase in ['zero','technical']:
        write(guard_dir/(phase+'_guard_release.template.json'),dict(reviewer='root',external_guard_authorized=False,
            policy_sha256=sha(guard_dir/'policy.json'),authorized_phases=[phase],**({'zero_verification_sha256':None}if phase=='technical'else{})))
    all_bound[R+'/external_guard_below_floor_v1/guardian.py']=sha(guard_dir/'guardian.py')
    all_bound[R+'/external_guard_below_floor_v1/policy.json']=sha(guard_dir/'policy.json')
    write(P/'root_review_TEMPLATE.json',dict(reviewer='root',below_floor_technical_authorized=False,
        below_floor_production_authorized=False,physical_export_authorized=False,
        spec_sha256=spec_hashes,controller_plan_sha256=sha(P/'technical_plan.json'),
        per_spec_releases=release_templates,bound_files=all_bound,
        source_RGB_and_geometry_before_update_gate='zero_source full seven states plus original all3FHD RGB/alpha gates',
        source_zero_controller_plan_sha256=sha(P/'zero_plan.json'),
        external_guard_policy_sha256=sha(guard_dir/'policy.json'),
        no_production_plan=True,no_automatic_progression=True))
    print(json.dumps(dict(package=str(P),worker_sha256=sha(worker),CUDA_started=False)))


if __name__=='__main__':main()
