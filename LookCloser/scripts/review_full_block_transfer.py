"""Receive hash-verified paired fusion and compare current hard-source RGB.

Raw per-view/full-block fusion is a causal control, NOT automatic replacement
of the production mesh with its later silhouette/head repairs.
"""
import argparse
from copy import deepcopy
from pathlib import Path, PurePosixPath
import subprocess
import json
import numpy as np
import open3d as o3d
from PIL import Image
from joint_temporal_texture import read, sha, atomic_json, cameras
from review_temporal_full_block_control import transfer_mesh_gauge
from review_jaw_repair_transfer import verified_image, panel

ROOT = Path('/mnt/data/dec5_full_block_transfer')
REMOTE = Path('/fsx/oregon/dec5_full_block_transfer')
VIDEO = Path('/mnt/data/dec5_large_motion_choices_v3/left_high_arc')


def retained_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Unsafe retained artifact path')
    return root.joinpath(*relative.parts)


def receive(frame):
    root = ROOT/frame
    text = subprocess.check_output(['ssh','ubuntu@dev3','cat',str(REMOTE/frame/'complete.json')], text=True)
    receipt = json.loads(text)
    assert receipt['request_sha256'] == sha(root/'request.json')
    assert set(receipt['hashes']) == {'depth_qc.json','binary.json','commands.json',
        'fuse-original/mesh.ply','fuse-original/mesh.json','fuse-full-block/mesh.ply','fuse-full-block/mesh.json'}
    for name in receipt['hashes']:
        target = retained_path(root,name); target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['rsync','-r',f'ubuntu@dev3:{REMOTE/frame/name}',str(target)],check=True)
        assert sha(target) == receipt['hashes'][name]
    for name in ('stages','logs'):
        subprocess.run(['rsync','-r',f'ubuntu@dev3:{REMOTE/frame/name}/',str(root/name)+'/'],check=True)
    subprocess.run(['rsync','-r',f'ubuntu@dev3:{REMOTE/frame}/checks.jsonl',str(root/'remote_checks.jsonl')],check=True)
    target=root/'pipeline/dense/stereo/depth_maps'; target.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-r','--include=*/','--include=*.geometric.bin','--exclude=*',
        f'ubuntu@dev3:{REMOTE/frame}/pipeline/dense/stereo/depth_maps/',str(target)+'/'],check=True)
    qc=read(root/'depth_qc.json'); assert qc['maps']==62 and qc['shape']==[1080,1920]
    for name,digest in qc['hashes'].items():
        path=retained_path(root,name)
        assert path.is_relative_to(target) and path.name.endswith('.geometric.bin')
        assert sha(path)==digest
    assert len(list(target.glob('**/*.geometric.bin')))==62
    atomic_json(root/'remote_complete.json',receipt)
    atomic_json(root/'real_depth_input.json',dict(dense=str(root/'pipeline/dense'),transforms=str(root/'data/transforms.json')))
    atomic_json(root/'received.json',dict(remote_complete_sha256=sha(root/'remote_complete.json'),
        request_sha256=sha(root/'request.json'),retained_hashes=receipt['hashes'],
        depth_hashes=qc['hashes'],all_retained_bytes_verified=True,remote_scratch_deleted=False))
    print(frame,'received two meshes and62 raw geometric maps; all hashes verified',flush=True)


def prepare(frame):
    root=ROOT/frame; received=read(root/'received.json')
    assert received['request_sha256']==sha(root/'request.json')
    for name,digest in {**received['retained_hashes'],**received['depth_hashes']}.items(): assert sha(root/name)==digest
    from study_confidence_depth_prior import load_real
    _,depths,_=load_real(ROOT,frame)
    assert len(depths)==62
    for depth in depths: assert depth.shape==(1080,1920) and np.isfinite(depth).all() and (depth>0).any()
    coverage=[float((d>0).mean()) for d in depths];qc=read(root/'depth_qc.json')
    np.testing.assert_allclose([np.mean(coverage),min(coverage)],[qc['coverage_mean'],qc['coverage_min']],rtol=0,atol=1e-12)
    del depths
    parent=read(VIDEO/'request.json'); entry=next(r for r in parent['inventory'] if r['frame_id']==frame)
    assert sha(entry['mesh'])==entry['mesh_sha256'] and sha(entry['metadata'])==entry['metadata_sha256']
    target_meta=read(entry['metadata']); rows,_,_=cameras(frame)
    # Resolve by rig prefix to avoid assuming a physical serial identifier.
    native=[next(r for r in rows if r['physical_camera'].startswith(prefix)) for prefix in ('H004_C005','K004_B005')]
    aligned={}
    for variant in ('fuse-original','fuse-full-block'):
        mesh=o3d.io.read_triangle_mesh(str(root/variant/'mesh.ply')); meta=read(root/variant/'mesh.json')
        old=np.asarray(mesh.vertices).copy(); new=transfer_mesh_gauge(old,meta,target_meta)
        np.testing.assert_allclose(transfer_mesh_gauge(new,target_meta,meta),old,rtol=0,atol=1e-12)
        mesh.vertices=o3d.utility.Vector3dVector(new)
        out=root/'aligned'/variant;out.mkdir(parents=True,exist_ok=True)
        path=out/'mesh.ply';o3d.io.write_triangle_mesh(str(path),mesh)
        aligned[variant]=dict(mesh=str(path),mesh_sha256=sha(path),original_mesh_sha256=sha(root/variant/'mesh.ply'),
            source_metadata_sha256=sha(root/variant/'mesh.json'),target_metadata_sha256=entry['metadata_sha256'],
            max_gauge_displacement=float(np.linalg.norm(old-new,axis=1).max()),
            roundtrip_passed=True,shape_changed_by_gauge_transfer=False)
        atomic_json(out/'result.json',aligned[variant])
    targets=[('moving',entry['camera'])]+[(r['physical_camera'],r) for r in native]
    for view,camera in targets:
        for arm in ('production','fuse-original','fuse-full-block'):
            if view=='moving' and arm=='production': continue
            request=deepcopy(parent); row=deepcopy(entry); target=deepcopy(camera)
            if view!='moving':
                target['physical_camera']='diagnostic_unmasked_target_'+view
                target['reference_physical_camera']=view
            row['camera']=target
            if arm!='production': row.update(mesh=aligned[arm]['mesh'],mesh_sha256=aligned[arm]['mesh_sha256'])
            request['inventory']=[row];request['ordered_frame_ids']=[frame]
            request['source_rows']=[r for r in request['source_rows'] if Path(r['source_dataset']).name==frame]
            request.update(partial_diagnostic_only=True,full_video_candidate=False,artifact_free_approval=False,
                full_block_transfer_arm=arm,geometry_changed=arm!='production',
                controlled_raw_fusion_pair=arm!='production',production_head_repairs_reapplied=False,
                texture_source_masks_unchanged=True,native_target_mask_disabled=view!='moving',
                paired_depth_qc_sha256=sha(root/'depth_qc.json'),paired_received_sha256=sha(root/'received.json'))
            for name in (Path(__file__).name,'review_temporal_full_block_control.py'):
                request['script_hashes'][name]=sha(Path(__file__).with_name(name))
            out=root/'rgb'/view/arm; (out/'frames').mkdir(parents=True,exist_ok=True)
            if (out/'request.json').exists(): assert read(out/'request.json')==request
            else:atomic_json(out/'request.json',request)
    atomic_json(root/'review_request.json',dict(frame=frame,views=[x[0] for x in targets],aligned=aligned,
        source_video_request_sha256=sha(VIDEO/'request.json'),script_sha256=sha(__file__),production_changed=False))
    print(frame,'three-view comparisons prepared; not rendered',flush=True)


def render(frame,view):
    from run_view_consistent_dynamic_video import install
    import render_smooth_temporal_mesh_video as engine
    implementation=install();engine.torch.set_num_threads(2)
    for arm in ('production','fuse-original','fuse-full-block'):
        if view=='moving' and arm=='production': continue
        out=ROOT/frame/'rgb'/view/arm;q=read(out/'request.json')
        assert implementation==q['source_quality_implementation_sha256']
        engine.render(out,[frame])


def review(frame):
    from calibrated_depth_witness import load_images
    root=ROOT/frame;q=read(root/'review_request.json'); images,_,source_receipt=load_images(frame)
    records=[]
    for view in q['views']:
        rgb={};receipts={};requests={};depths={}
        for arm in ('production','fuse-original','fuse-full-block'):
            folder=VIDEO if arm=='production' and view=='moving' else root/'rgb'/view/arm
            rgb[arm],receipts[arm]=verified_image(folder,frame);requests[arm]=read(folder/'request.json')
            depths[arm]=np.load(folder/'frames'/frame/'target_depth.npz')['depth']
        for arm in rgb:
            for key in ('camera','source_cameras','fixed_exposure'): assert receipts[arm][key]==receipts['production'][key]
            for key in ('profiles_sha256','exposure_sha256','calibration_sha256'): assert requests[arm][key]==requests['production'][key]
        im=[rgb[x] for x in rgb];labels=['production (later repairs)','raw per-view fusion','raw full-block fusion']
        if view!='moving': im.insert(0,np.rot90(images[view]));labels.insert(0,'real train GT')
        out=root/'review'/view;out.mkdir(parents=True,exist_ok=True)
        if view!='moving': Image.fromarray(np.rot90(images[view])).save(out/'train_gt.png')
        for name,box in [('head',(100,350,1000,1150)),('hand',(150,900,850,1600))]:
            panel(out/(name+'.png'),im,labels,box)
            for arm,image in rgb.items(): Image.fromarray(image).crop(box).save(out/(arm+'_'+name+'.png'))
        before=depths['fuse-original'];after=depths['fuse-full-block']
        records.append(dict(view=view,geometry_new_depth=int(((before==0)&(after>0)).sum()),
            geometry_lost_depth=int(((before>0)&(after==0)).sum()),
            changed_pair_rgb=int(np.any(rgb['fuse-original']!=rgb['fuse-full-block'],axis=2).sum()),
            request_hashes={a:sha((VIDEO if a=='production' and view=='moving' else root/'rgb'/view/a)/'request.json') for a in rgb},
            image_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()},visual_status='pending'))
    atomic_json(root/'review/result.json',dict(records=records,source_receipt=source_receipt,
        novel_view_gt_available=False,counts_not_quality_metrics=True,production_promoted=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('receive','prepare','render','review'))
    parser.add_argument('--frame',required=True,choices=('000995','000997'));parser.add_argument('--view')
    a=parser.parse_args()
    if a.action=='render':
        if a.view is None: parser.error('render requires --view')
        render(a.frame,a.view)
    else:globals()[a.action](a.frame)
