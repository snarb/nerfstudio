"""Seal a matched learned128/refinement stage without authorizing GPU work."""
import argparse
import json
from pathlib import Path
import shutil
from prepare_marshawn_frozen_semantic_cpu import sha, write


def prepare(campaign, output):
    if output.exists():
        raise FileExistsError('Use a fresh experiment namespace')
    remote = '/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106'
    def r(path):
        return str(path).replace(str(campaign), remote)
    output.mkdir(parents=True); (output/'code').mkdir(); (output/'specs').mkdir(); (output/'controllers').mkdir()
    dependencies = ['probe_marshawn_LC_semantic_refinement_head.py', 'marshawn_LC_semantic_refinement_contract.py',
                    'marshawn_semantic_residual_head.py', 'audit_marshawn_LC_semantic_refinement_cpu.py',
                    'launch_marshawn_LC_semantic_refinement.py', 'compare_marshawn_LC_semantic_refinement_technical_cpu.py', 'marshawn_frozen_semantic_head.py',
                    'prepare_marshawn_frozen_semantic_cpu.py', 'run_marshawn_frozen_semantic_probe.py',
                    'marshawn_gpu_isolation.py']
    for name in dependencies:
        shutil.copyfile(Path(__file__).with_name(name), output/'code'/name)
    shutil.copyfile(campaign/'frozen_LC_semantic_resolution256_fourUNK_plan_v1/code/marshawn_LC_semantic_resolution_contract.py',
                    output/'code/marshawn_LC_semantic_resolution_contract.py')
    witnesses = campaign/'frozen_LC_semantic_resolution256_fourUNK_plan_v1/CPU_witnesses109.json'
    shutil.copyfile(witnesses, output/'CPU_witnesses109.json')
    bound = {r(p): sha(p) for p in (output/'code').iterdir()}
    for p in [output/'CPU_witnesses109.json',
              campaign/'frozen_LC_semantic_head_silver_alias_v1/head_input_contract.json',
              campaign/'frozen_LC_semantic_head_silver_alias_v1/capture_adapter_receipt.json',
              campaign/'frozen_LC_semantic_LA_two_UNKNOWN_candidate_v1/label_qualification.json',
              campaign/'frozen_native_semantic_prior001_candidate_v1/camera_history_reference.json']:
        bound[r(p)] = sha(p)
    base = remote+'/frozen_LC_semantic_LA_two_UNKNOWN_candidate_v1/runs/prior001_fresh2064/head.pt'
    from marshawn_LC_semantic_refinement_contract import BASE_SHA, LABEL_SHA, FAMILY, ARMS
    bound[base] = BASE_SHA
    controller=campaign/'frozen_LC_semantic_capture_plan_v4/code/marshawn_tuning_controller.py'
    bound[r(controller)]=sha(controller)
    source=json.loads((campaign/'frozen_LC_semantic_head_silver_alias_v1/head_input_contract.json').read_text())
    bound[source['native_source']['checkpoint']]=source['native_source']['checkpoint_sha256']
    plans = {}
    for stage in ['technical', 'first2064_gate']:
        jobs = []
        for arm, gpu in [('coarse128', 3), ('residual256', 4)]:
            parent_name = None
            phases = [('technical32',32,None), ('technical_resume32',32,'technical32'), ('technical64',64,None)] if stage == 'technical' else [('fresh2064',2064,None)]
            for phase, updates, resume in phases:
                name = arm+'_'+phase; specp = output/'specs'/(name+'.json')
                spec = dict(phase='direct_LC_refinement_'+name, arm=arm, head_type=ARMS[arm], head_schema_version=1,
                    refinement_family=FAMILY, root_release=r(output/'root_release.json'),
                    manifest=remote+'/frozen_LC_semantic_head_silver_alias_v1/head_input_contract.json',
                    capture_receipt=remote+'/frozen_LC_semantic_head_silver_alias_v1/capture_adapter_receipt.json',
                    label_qualification=remote+'/frozen_LC_semantic_LA_two_UNKNOWN_candidate_v1/label_qualification.json',
                    label_qualification_sha256=LABEL_SHA, base_checkpoint=base, base_checkpoint_sha256=BASE_SHA,
                    stage_optimizer_policy='EXPLICIT_NEW_STAGE_RESET_ADAM0_AND_SAMPLER_SEED42',
                    capacity_policy='KEEP_GENUINE_LABELS_AND_REPORT_IRREDUCIBLE_BOUND',
                    authorization_key='LC_refinement_technical_head_authorized' if stage == 'technical' else 'LC_refinement_first2064_head_authorized',
                    initial_probability=.01, updates=updates, torch_version='2.7.1+cu128', cuda_version='12.8',
                    output=r(output/'runs'/name), seed=42, learning_rate=.03, native_model_updates=0, qualified_UNKNOWN_count=4,
                    packet_history_reference=remote+'/frozen_native_semantic_prior001_candidate_v1/camera_history_reference.json',
                    camera_history_reference=remote+'/frozen_native_semantic_prior001_candidate_v1/camera_history_reference.json',
                    technical_gate_required_before_fresh2064=True,
                    controlled_change='Parameterization only at the SAME learned128 basefunction; matched explicit new Adam0/LR.03/seed42 refinement stage in BOTH arms.')
                if resume:
                    spec.update(resume=r(output/'runs'/(arm+'_'+resume)/'head.pt'), resume_initial_iteration=32)
                write(specp,spec)
                jobname='LC_refinement_'+name
                jobs.append(dict(name=jobname,host='silver'+str(gpu), command=[remote+'/python',r(output/'code/probe_marshawn_LC_semantic_refinement_head.py'),'--spec',r(specp),'--execute'],
                    cwd=r(output/'code'),env=dict(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                        LOOKCLOSER_GPU_LEASE_RECEIPT='{job_root}/receipt.json', PYTHONUNBUFFERED='1'),
                    dependencies=[parent_name] if parent_name else [],poll_seconds=5,
                    progress_files=[spec['output']+'/progress.json'],metrics_file=spec['output']+'/complete.json'))
                parent_name=jobname
        plan=dict(schema_version=1,poll_seconds=15,hosts={
            'silver'+str(g):dict(address='ubuntu@silver-sable',python=remote+'/python',
                controller=remote+'/frozen_LC_semantic_capture_plan_v4/code/marshawn_tuning_controller.py',
                root=r(output/'queue'/stage/('gpu'+str(g))),gpus=[g]) for g in [3,4]},jobs=jobs)
        pp=output/'controllers'/(stage+'.json');write(pp,plan);plans[stage]=sha(pp)
    specs = {r(p):sha(p) for p in (output/'specs').glob('*.json')}
    release=dict(reviewer=None, LC_refinement_technical_head_authorized=False, LC_refinement_first2064_head_authorized=False,
        LC_refinement_controller_technical_authorized=False,LC_refinement_controller_first2064_authorized=False,
        spec_sha256=specs,controller_plan_sha256=plans,bound_files=bound,label_qualification_sha256=LABEL_SHA,
        mandatory_gate='ROOT technical acceptance before fresh2064. New head schema required for physical exports.')
    write(output/'root_release_TEMPLATE.json',release)
    write(output/'CPU_readiness.json',dict(CPU_only=True,CUDA_started=False,authorized=False,root=r(output),
        python=remote+'/python',spec_count=len(specs),controller_plan_sha256=plans,
        dispatcher_host='local_olive',dispatcher_python='/home/pavlo/.conda/envs/nerfstudio/bin/python',
        dispatcher_controller=str(campaign/'frozen_LC_semantic_capture_plan_v4/code/marshawn_tuning_controller.py'),
        remote_workers_only=True,self_SSH_supported=False,
        GPUs={'SG3':'GPU-c024b6e2-0def-f4fd-c18a-64b95975effd','SG4':'GPU-cb1a0b07-abd6-8e55-9669-ccbb2c45a560'},
        source_checkpoint_sha256=BASE_SHA, matched_refinement_stage_reset=True,
        original2064_control_reused_separately=True, physical_renderer_integrated=False))
    print(json.dumps(dict(output=str(output),spec_count=len(specs),CUDA_started=False,authorization=False)))

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.campaign,a.output)
