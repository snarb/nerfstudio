"""Seal a bounded unchanged-observer witness and adversarial CPU arithmetic proof."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from prepare_marshawn_frozen_semantic_cpu import sha,write


def prepare(p2,output):
    if output.exists():raise FileExistsError('New immutable witness namespace required')
    original=json.loads((p2/'release_TEMPLATE.json').read_text());bound=dict(original['bound_files'])
    for p,digest in bound.items():
        if sha(p)!=digest:raise ValueError('P2 source changed: '+p)
    observer=p2/'code/marshawn_LC_semantic_physical_path.py'
    if sha(observer)!='175274175c298676823c08b2d473a30b17bc8be34057a79f47f0f045e25521f5':raise ValueError('Original P2 observer required')
    output.mkdir(parents=True);(output/'code').mkdir()
    for name in ['witness_marshawn_LC_density_weights.py','marshawn_stable_packed_density_reference.py','prepare_marshawn_frozen_semantic_cpu.py']:
        target=output/'code'/name;shutil.copyfile(Path(__file__).with_name(name),target);bound[str(target)]=sha(target)
    controller=output/'code/marshawn_tuning_controller.py';shutil.copyfile(p2/'code/marshawn_tuning_controller.py',controller);bound[str(controller)]=sha(controller)
    spec=json.loads((p2/'specs/uncapped_qualification.json').read_text())
    spec.update(root_release=str(output/'root_release.json'),output=str(output/'witness'),phase='unchanged_P2_weight_reference_witness',
        path_mode='native1024',camera='DC',first_pixel=1409152,last_pixel_exclusive=1425536,chunk=128,max_seconds=600,
        P2_code=str(p2/'code'),P2_observer=str(observer),P2_observer_sha256=sha(observer))
    path=output/'spec.json';write(path,spec)
    m=json.loads(Path(spec['LC_source_plan']).read_text())
    job=dict(name='unchanged_P2_native1024_weight_witness',host='olive',command=[m['python'],str(output/'code/witness_marshawn_LC_density_weights.py'),
        '--spec',str(path),'--execute'],cwd=str(output/'code'),env=dict(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        PYTHONUNBUFFERED='1',EXPECTED_GPU_UUID=m['expected_physical_UUID'],LOOKCLOSER_GPU_LEASE_RECEIPT='{job_root}/receipt.json'),
        dependencies=[],poll_seconds=5,progress_files=[str(output/'witness/progress.json')],metrics_file=str(output/'witness/complete.json'))
    controller_path=output/'controller_plan.json'
    write(controller_path,dict(schema_version=1,poll_seconds=15,hosts=dict(olive=dict(python=m['python'],controller=str(controller),
        root=str(output/'queue'),gpus=[0])),jobs=[job]))
    for p in [p2/'uncapped_failure_receipt.json',p2/'uncapped_qualification/progress.json',
            p2/'queue/uncapped_qualification/jobs/uncapped_source_qualification/worker.log',
            p2/'uncapped_qualification/source_FHD_parity/parity.json']:
        bound[str(p)]=sha(p)
    write(output/'release_TEMPLATE.json',dict(reviewer='root',LC_weight_witness_authorized=False,
        controller_plan_sha256=sha(controller_path),spec_sha256={str(path):sha(path)},bound_files=bound,
        native_updates=0,head_updates=0,stable_reference_replacement_authorized=False,automatic_recovery_or_render_continuation=False))
    import torch
    sys.path.insert(0,str(output/'code'))
    from marshawn_stable_packed_density_reference import stable_packed_density_weights
    loader=importlib.util.spec_from_file_location('_CPU_unchanged_P2_reference',observer);legacy=importlib.util.module_from_spec(loader);loader.loader.exec_module(legacy)
    cases=[]
    for name,density,ids,ray_count in [('cross_ray',[2.**54,.25,.5],[0,1,1],2),('current_tau_subtraction',[.25,2.**54],[0,0],1)]:
        sigma=torch.tensor(density,dtype=torch.float32);starts=torch.zeros_like(sigma);ends=torch.ones_like(sigma);ids=torch.tensor(ids)
        optical_before=torch.zeros(ray_count,dtype=torch.float64);serial=[]
        for s,e,d,i in zip(starts,ends,sigma,ids):
            tau=d.double()*(e.double()-s.double());serial.append(torch.exp(-optical_before[i])*-torch.expm1(-tau));optical_before[i]+=tau
        serial=torch.stack(serial);old=legacy.packed_density_weights(starts,ends,sigma,ids,ray_count);stable=stable_packed_density_weights(starts,ends,sigma,ids,ray_count)
        if (old-serial).abs().max()<=.01 or not torch.allclose(stable,serial,atol=1e-15,rtol=1e-14):raise ValueError('CPU adversarial reference proof failed')
        cases.append(dict(name=name,input_sigma_float32=sigma.tolist(),old=old.tolist(),serial_FP64=serial.tolist(),stable=stable.tolist(),
            old_max_error=float((old-serial).abs().max()),stable_max_error=float((stable-serial).abs().max())))
    write(output/'CPU_adversarial_proof.json',dict(CPU_only=True,CUDA_initialized=torch.cuda.is_initialized(),optimizer_created=False,
        unchanged_P2_observer_sha256=sha(observer),stable_helper_sha256=sha(output/'code/marshawn_stable_packed_density_reference.py'),cases=cases,
        actual_failed_batch_not_yet_observed=True,stable_reference_not_accepted=True,
        interpretation='These adversarial cases prove old-reference cancellation mechanisms, not the cause of the actual P2 failure. Actual unchanged-P2 CUDA witness and CPU comparison are required before accepting any reference-only recovery.'))
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--P2',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.P2,args.output)
