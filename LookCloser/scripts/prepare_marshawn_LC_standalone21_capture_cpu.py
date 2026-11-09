"""Prepare an independent false-release21+7 native-source diagnostic package."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

C=Path('/fsx/tmp/marshawn_001660_hd/tuning_20261106')
SC='/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106'
NAME='frozen_LC_standalone21_capture_plan_v1';P=C/NAME;R=SC+'/'+NAME
BASE=C/'frozen_LC_semantic_source_silver_parity_pathfix_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def remote(p):return str(p).replace(str(C),SC)


def main():
    sys.path.insert(0,str(C/'semantic_quarter_distortion_candidate_20261107/code'))
    import torch
    from nerfstudio.cameras.cameras import Cameras,CameraType
    if torch.cuda.is_initialized():raise ValueError('CPU-only preparation')
    P.mkdir(exist_ok=True);(P/'queries').mkdir(exist_ok=True);(P/'code').mkdir(exist_ok=True)
    for name in ['marshawn_LC_frozen_source.py','marshawn_gpu_isolation.py',
                 'marshawn_tuning_controller.py','prepare_marshawn_frozen_semantic_cpu.py']:
        shutil.copyfile(BASE/'code'/name,P/'code'/name)
    for name in ['marshawn_LC_semantic_physical_path.py','marshawn_stable_packed_density_reference.py']:
        shutil.copyfile(C/'frozen_LC_semantic_physical_plan_v3/code'/name,P/'code'/name)
    worker=Path(__file__).with_name('run_marshawn_LC_standalone21_capture.py');shutil.copyfile(worker,P/'code'/worker.name)
    extra=C/'native_actor_extra16_capture_plan_CPU_v1/manifest.json'
    if sha(extra)!='9205c0953dab3bc7b3710e1b750dc95b32f26f540df26daa1cd42061b20c721a':raise ValueError('Exact extra16 proposal changed')
    em=json.loads(extra.read_text());original=C/'frozen_native_semantic_CPU_v2/manifest.json';om=json.loads(original.read_text())
    frames={f['physical_camera']:f for f in json.loads((C/'data/transforms.json').read_text())['frames']}
    rows={};provenance=[]
    for record in em['records']:
        file=Path(record['archive'])
        if sha(file)!=record['archive_sha256']:raise ValueError('Extra16 changed')
        with np.load(file,allow_pickle=False)as z:
            for i in range(record['rays']):
                rows.setdefault(record['camera'],[]).append(dict(
                    query_id=str(z['query_id'][i]),pixel_xy=z['pixel_xy'][i].copy(),
                    semantic_target=float(z['semantic_target'][i]),review_confidence=float(z['review_confidence'][i]),
                    proposed_future_weight=float(z['proposed_post_qualification_label_weight'][i]),
                    category=str(z['category'][i]),original_global_query_index=-1,
                    origin_namespace='extra16_source_review',source_record_index=int(z['standalone_row_index'][i])))
    wanted={66502,66665,66496,61101,61499};found=set();offset=0
    for record in om['records']:
        file=Path(record['archive'])
        if sha(file)!=record['archive_sha256']:raise ValueError('Original63 query archive changed')
        with np.load(file,allow_pickle=False)as z:
            n=len(z['pixel_xy'])
            for g in sorted(wanted):
                if offset<=g<offset+n:
                    i=g-offset;found.add(g);rows.setdefault(record['camera'],[]).append(dict(
                        query_id='existing_global_'+str(g),pixel_xy=z['pixel_xy'][i].astype(np.float64),
                        semantic_target=float(z['semantic_target'][i]),review_confidence=float(z['label_weight'][i]),
                        proposed_future_weight=0.,category=str(z['category'][i]),original_global_query_index=g,
                        origin_namespace='existing_frozen_69888_diagnostic_replay',source_record_index=i))
                    provenance.append(dict(global_query_index=g,camera=record['camera'],camera_local_index=i,
                        pixel_xy=z['pixel_xy'][i].tolist(),original_target=float(z['semantic_target'][i]),
                        original_label_weight=float(z['label_weight'][i]),source_archive=record['archive'],source_archive_sha256=record['archive_sha256']))
        offset+=n
    if found!=wanted or offset!=69888:raise ValueError('Exact five IDs or total membership differs')
    records=[];allrows=[]
    for cam,selected in sorted(rows.items()):
        f=frames[cam];camera=Cameras(camera_to_worlds=torch.tensor(f['transform_matrix'],dtype=torch.float32)[None,:3],
            fx=f['fl_x'],fy=f['fl_y'],cx=f['cx'],cy=f['cy'],width=f['w'],height=f['h'],camera_type=CameraType.PERSPECTIVE)
        xy=np.array([q['pixel_xy']for q in selected],np.float64)
        rays=camera.generate_rays(0,coords=torch.tensor(xy[:,[1,0]]+.5,dtype=torch.float32),keep_shape=False)
        for q in selected:q['pixel_xy']=q['pixel_xy'].tolist()
        dest=P/'queries'/(cam+'.npz');n=len(selected)
        np.savez_compressed(dest,query_id=np.array([q['query_id']for q in selected]),pixel_xy=xy,
            coords_yx_COLMAP=xy[:,[1,0]]+.5,origins=rays.origins.numpy(),directions=rays.directions.numpy(),
            pixel_area=rays.pixel_area.numpy(),directions_norm=rays.metadata['directions_norm'].numpy(),
            semantic_target=np.array([q['semantic_target']for q in selected],np.float32),
            label_weight=np.zeros(n,np.float32),review_confidence=np.array([q['review_confidence']for q in selected],np.float32),
            proposed_future_weight=np.array([q['proposed_future_weight']for q in selected],np.float32),
            category=np.array([q['category']for q in selected]),
            original_global_query_index=np.array([q['original_global_query_index']for q in selected],np.int64),
            world_to_box=np.eye(4))
        records.append(dict(camera=cam,rays=n,archive=remote(dest),archive_sha256=sha(dest)))
        allrows.extend(dict(camera=cam,**q)for q in selected)
    if sum(q['rays']for q in records)!=21:raise ValueError('Exactly21queries required')
    cells=C/'semantic_cell_candidates_cpu_v1/component_floor_inventory_v1/active_r0_low_cells.json'
    if len(json.loads(cells.read_text()))!=7:raise ValueError('Seven declared native cells required')
    shutil.copyfile(cells,P/'seven_low_cells.json')
    manifest=dict(kind='STANDALONE21_TRAIN_QUERY_AND7_POINT_DENSITY_DIAGNOSTIC',records=records,rows=allrows,
        extra16_canonical_manifest=str(extra),extra16_canonical_manifest_sha256=sha(extra),
        original_queries_manifest_sha256=sha(original),original_69888_rays_unchanged=True,
        existing_five_provenance=provenance,source_pipeline_digest=em['source_pipeline_digest'],
        source_checkpoint_sha256=em['source_checkpoint_sha256'],world_to_box=np.eye(4).tolist(),
        current_fit_weights_all0=True,FD_subpixel_HD_footprint_pure_diagnostic=True,
        batching='One deterministic camera packet FD5/MC5/MD8/ND3. Existing5 original128-ray allocator context is not reused or claimed identical.',
        low_cells_source=str(cells),low_cells_source_sha256=sha(cells),point_sigma_not_visibility=True)
    write(P/'query_manifest.json',manifest)
    capture=C/'frozen_LC_semantic_capture_plan_v4/capture/complete.json';cm=json.loads(capture.read_text())
    cache={remote(q['archive']):q['archive_sha256']for q in cm['records']if 'archive'in q}
    spec=dict(phase='standalone21_native_TRAIN_capture_and7_point_density',output=R+'/capture',root_release=R+'/root_release.json',
        LC_source_plan=remote(BASE/'source_execution_manifest.json'),LC_source_release=remote(BASE/'source_loader_release.json'),
        LC_source_worker_name='run_marshawn_lc_semantic_capture_v3.py',query_manifest=R+'/query_manifest.json',
        low_cells=R+'/seven_low_cells.json',transforms=SC+'/data/transforms.json',max_seconds=1800,original_cache_files=cache,
        original_LC_capture_receipt_sha256=sha(capture),source_parity_completion_sha256=sha(BASE/'parity/complete.json'),
        native_path='original_legacy_ARM1024',field_sigma='unchanged',head_loaded=False,optimizer_authorized=False)
    write(P/'spec.json',spec)
    previous=json.loads((BASE/'root_release.json').read_text());bound=dict(previous['bound_files']);bound.update(cache)
    for file in [extra,original,capture,BASE/'parity/complete.json',C/'data/transforms.json']:
        bound[remote(file)]=sha(file)
    for file in [P/'query_manifest.json',P/'seven_low_cells.json',P/'spec.json',*(P/'queries').glob('*.npz'),*(P/'code').glob('*.py')]:
        bound[remote(file)]=sha(file)
    d=json.loads((BASE/'controller_plan.json').read_text());d['hosts']['silver'].update(controller=R+'/code/marshawn_tuning_controller.py',root=R+'/queue')
    j=d['jobs'][0];j.update(name='standalone21_and7_capture',cwd=R+'/code',metrics_file=R+'/capture/complete.json')
    j['command']=[SC+'/python',R+'/code/'+worker.name,'--spec',R+'/spec.json','--execute'];write(P/'controller_plan.json',d)
    bound[R+'/controller_plan.json']=sha(P/'controller_plan.json')
    write(P/'root_review_TEMPLATE.json',dict(reviewer='root',LC_standalone21_capture_authorized=False,
        head_fit_authorized=False,physical_selector_render_authorized=False,optimizer_authorized=False,
        spec_sha256={R+'/spec.json':sha(P/'spec.json')},controller_plan_sha256=sha(P/'controller_plan.json'),bound_files=bound))
    print(json.dumps(dict(package=str(P),records=records,bindings=len(bound),CUDA_started=False)))


if __name__=='__main__':main()
