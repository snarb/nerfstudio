"""CPU ray-footprint, original-ID, executable-closure and fail-closed checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def main():
    p=Path(sys.argv[1]);m=json.loads((p/'query_manifest.json').read_text());s=json.loads((p/'spec.json').read_text())
    policy=json.loads((p/'root_review_TEMPLATE.json').read_text())
    digest=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    bad=[f for f,h in policy['bound_files'].items()if not Path(f).is_file()or digest(f)!=h]
    if bad:raise ValueError('Actual executable closure mismatch:'+str(bad))
    source_plan=Path(s['LC_source_plan']);sm=json.loads(source_plan.read_text())
    sys.path[:0]=[str(source_plan.parent/'code')]
    import run_marshawn_lc_semantic_capture_v3 as capture_worker
    capture_worker.authorize(source_plan,Path(s['LC_source_release']),Path(sm['campaign']))
    if 'torch'in sys.modules:raise ValueError('Pre-model source authorizer importedTorch')
    sys.path.insert(0,sm['branch']+'/code')
    import torch
    from nerfstudio.cameras.cameras import Cameras,CameraType
    frames={f['physical_camera']:f for f in json.loads(Path(s['transforms']).read_text())['frames']}
    ids=[];originals=[];max_errors={k:0. for k in ['origins','directions','pixel_area','directions_norm']}
    rows=[]
    for record in m['records']:
        with np.load(record['archive'],allow_pickle=False)as z:q={k:z[k].copy()for k in z.files}
        n=len(q['pixel_xy']);assert n==record['rays'];assert np.all(q['label_weight']==0)
        assert np.array_equal(q['coords_yx_COLMAP'],q['pixel_xy'][:,[1,0]]+.5)
        assert np.array_equal(q['world_to_box'],np.eye(4))
        f=frames[record['camera']];camera=Cameras(camera_to_worlds=torch.tensor(f['transform_matrix'],dtype=torch.float32)[None,:3],
            fx=f['fl_x'],fy=f['fl_y'],cx=f['cx'],cy=f['cy'],width=f['w'],height=f['h'],camera_type=CameraType.PERSPECTIVE)
        rays=camera.generate_rays(0,coords=torch.tensor(q['coords_yx_COLMAP'],dtype=torch.float32),keep_shape=False)
        for key,actual in [('origins',rays.origins),('directions',rays.directions),('pixel_area',rays.pixel_area),('directions_norm',rays.metadata['directions_norm'])]:
            error=float(abs(actual.numpy()-q[key]).max());max_errors[key]=max(max_errors[key],error)
            tolerance={'origins':2e-7,'directions':5e-7,'pixel_area':1e-10,'directions_norm':5e-7}[key]
            if error>tolerance:raise ValueError('CPU native camera ray footprint mismatch:'+key+' '+str(error))
        ids.extend(q['query_id'].tolist());originals.extend(q['original_global_query_index'][q['original_global_query_index']>=0].tolist())
        rows.append(dict(camera=record['camera'],rays=n,fit_weight_nonzero=int(np.count_nonzero(q['label_weight']))))
    if len(ids)!=21 or len(set(ids))!=21 or sorted(originals)!=[61101,61499,66496,66502,66665]:raise ValueError('Exact21 IDs changed')
    assert len(s['original_cache_files'])==60
    cells=json.loads(Path(s['low_cells']).read_text());assert len(cells)==7
    for cell in cells:
        assert np.array_equal(np.array(cell['center']),(.5*(np.array(cell['lo'])+np.array(cell['hi']))))
    code=p/'code';env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    dry=subprocess.run([sys.executable,str(code/'run_marshawn_LC_standalone21_capture.py'),'--spec',str(p/'spec.json')],env=env,capture_output=True,text=True)
    controller=subprocess.run([sys.executable,str(code/'marshawn_tuning_controller.py'),'dry-run','--root',str(p/'controller'),'--manifest',str(p/'controller_plan.json')],env=env,capture_output=True,text=True)
    false=p/'CPU_false_authorization_fixture.json';false.write_text(json.dumps(policy))
    fixture_spec=dict(s,root_release=str(false));fixture=p/'CPU_false_spec_fixture.json';fixture.write_text(json.dumps(fixture_spec))
    blocked=subprocess.run([sys.executable,str(code/'run_marshawn_LC_standalone21_capture.py'),'--spec',str(fixture),'--execute'],env=env,capture_output=True,text=True)
    if dry.returncode or controller.returncode or blocked.returncode==0 or 'Exact independent root21+7 capture release required'not in blocked.stderr:
        raise ValueError('Dryrun or fail-closed check failed')
    probe=p/'CPU_actual_write_probe';probe.mkdir();probe.rmdir()
    if torch.cuda.is_initialized():raise ValueError('CPU audit created CUDA context')
    result=dict(passed=True,bindings=len(policy['bound_files']),exact_source_authorizer_all63=True,TRAIN_records=rows,
        exact21_unique_ids=True,exact5_existing_ids=sorted(originals),all_fit_weights0=True,
        native_CPU_ray_errors=max_errors,seven_declared_low_cell_centers=True,original_cache_hashes60=True,
        dryruns=[dry.returncode,controller.returncode],false_release_failed_before_source_load=True,
        output_absent=not(p/'capture').exists(),actual_write_remove=True,CUDA_started=False,
        batching_caveat=m['batching'])
    (p/'CPU_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
