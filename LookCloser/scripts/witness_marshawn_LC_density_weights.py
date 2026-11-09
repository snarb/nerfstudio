"""Bounded read-only witness of the unchanged P2 native-weight proof failure."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
from prepare_marshawn_frozen_semantic_cpu import sha,write


def run(path,execute):
    spec=json.loads(path.read_text())
    if (spec['camera'],spec['first_pixel'],spec['last_pixel_exclusive'],spec['chunk'])!=('DC',1409152,1425536,128) or spec['path_mode']!='native1024':
        raise ValueError('Only exact root-requested DC witness interval is allowed')
    if not execute:
        print(json.dumps(dict(phase='unchanged_P2_native1024_weight_witness',pixels=[1409152,1425536],chunk=128,
            source_updates=0,optimizer_created=False,CUDA_started=False,stop_on_first_ValueError=True,
            unchanged_observer_sha256=spec['P2_observer_sha256'],reference_tolerance=2e-5)))
        return
    release=json.loads(Path(spec['root_release']).read_text())
    if release.get('reviewer')!='root'or release.get('LC_weight_witness_authorized')is not True or release['spec_sha256'].get(str(path))!=sha(path):
        raise ValueError('Exact separate root witness release required')
    for p,digest in release['bound_files'].items():
        if sha(p)!=digest:raise ValueError('Bound input changed: '+p)
    observer_path=Path(spec['P2_observer'])
    if sha(observer_path)!=spec['P2_observer_sha256']:raise ValueError('Unchanged P2 observer required')
    sys.path.insert(0,str(Path(spec['P2_code'])))
    from marshawn_gpu_isolation import require_controller_lease
    lease=require_controller_lease(os.environ['LOOKCLOSER_GPU_LEASE_RECEIPT'])
    import numpy as np
    import torch
    from marshawn_LC_frozen_source import load_source
    output=Path(spec['output']);output.mkdir(parents=True,exist_ok=False)
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Bounded native proof witness exceeded600seconds')))
    signal.alarm(600)
    source=load_source(spec,output,lease,release);model=source['model']
    loader=importlib.util.spec_from_file_location('_reviewed_lc_P2_unchanged_weight_observer',observer_path)
    adapter=importlib.util.module_from_spec(loader);sys.modules[loader.name]=adapter;loader.loader.exec_module(adapter)
    dataset=source['pipeline'].datamanager.eval_dataset
    names=[Path(p).stem.split('_')[-1]for p in dataset.image_filenames]
    camera=dataset.cameras[names.index('DC'):names.index('DC')+1].to(model.device)
    width=int(camera.width.item());rays=camera.generate_rays(0,keep_shape=True).flatten()
    result=None;completed=0
    with torch.inference_mode():
        for first in range(spec['first_pixel'],spec['last_pixel_exclusive'],128):
            rb=rays[first:first+128]
            try:
                row=adapter.capture_lc_path(model,rb,expected_cap=1024)
                del row
            except ValueError as error:
                frame=None;traceback=error.__traceback__
                while traceback is not None:
                    candidate=traceback.tb_frame
                    if candidate.f_code.co_name=='capture_lc_path'and Path(candidate.f_code.co_filename).resolve()==observer_path.resolve():
                        frame=candidate
                    traceback=traceback.tb_next
                required={'starts','ends','sigma','ids','w','rgb','xyz','native','proof_weights','errors'}
                if frame is None or not required<=frame.f_locals.keys():
                    write(output/'unexpected_failure.json',dict(first_pixel=first,error=repr(error),qualified_weight_frame_found=False))
                    raise
                values=frame.f_locals
                arrays={name:values[name].detach().cpu().numpy().copy()for name in ['starts','ends','sigma','ids','w','rgb','xyz','proof_weights']}
                arrays['native_rgb']=values['native']['rgb'].detach().cpu().numpy().copy()
                arrays['native_alpha']=values['native']['accumulation'].detach().cpu().numpy().copy()
                arrays['ray_origins']=values['rays'].origins.detach().cpu().numpy().copy()
                arrays['ray_directions']=values['rays'].directions.detach().cpu().numpy().copy()
                arrays['pixel_xy']=np.asarray([[(first+i)%width,(first+i)//width]for i in range(len(rb))],np.int32)
                archive=output/'unchanged_P2_actual_failure.npz';np.savez_compressed(archive,**arrays)
                difference=np.abs(arrays['w'].astype(np.float64)-arrays['proof_weights'])
                bad=np.flatnonzero(difference>2e-5)
                affected=sorted(set(arrays['ids'][bad].tolist()))
                result=dict(reproduced=True,first_pixel=first,last_pixel_exclusive=first+len(rb),
                    failing_sample_indices=bad.tolist(),failing_pixel_xy=arrays['pixel_xy'][affected].tolist(),
                    unchanged_observer_errors=values['errors'],caught_error=repr(error),
                    actual_failed_GPU_reference_saved=True,reference_or_tolerance_changed=False,
                    actual_native_weights_and_field_precision_unchanged=True,
                    archive=str(archive),archive_sha256=sha(archive))
                break
            completed+=len(rb)
            write(output/'progress.json',dict(completed_nonfailing_pixels=completed,current_first_pixel=first,
                source_updates=0,observer_unchanged=True,health=source['source_runner'].health(torch,lease['expected_physical_uuid'],source['manifest'])))
    checks=source['assert_unchanged']();source['guard_done'].set();signal.alarm(0)
    if result is None:result=dict(reproduced=False,examined_pixel_range=[1409152,1425536],no_scope_extension=True)
    write(output/'complete.json',dict(**result,completed_nonfailing_pixels=completed,source_unchanged=checks,
        source_checkpoint_sha256=source['manifest']['source_archive_sha256']['checkpoint.ckpt'],
        source_pipeline_digest=source['pipeline_digest'],unchanged_P2_observer_sha256=sha(observer_path),
        no_optimizer_created=True,no_head_loaded=True,sampler_native1024_unchanged=True,
        source_FHD_parity_passed=True,stopped_on_first_failure=bool(result['reproduced']),
        stable_reference_accepted=False,automatic_recovery_or_render_continuation=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();run(args.spec,args.execute)
