"""Checksum and paired-render audit; never infer a visual pass automatically."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from joint_temporal_texture import read, sha, atomic_json
from study_jaw_boundary_notches import PARENT


def run(root):
    records=[];panels=[]
    for frame in ['001193','001195']:
        control=root/'controls'/frame;complete=read(control/'complete.json')
        if complete['request_sha256']!=sha(control/'request.json'):raise ValueError('Control request changed')
        for path,digest in complete['hashes'].items():
            if sha(control/path)!=digest:raise ValueError('Control output changed')
        verified=0;superseded=[]
        for path in sorted((control/'stages').glob('*.json')):
            stage=read(path)
            if stage['request_sha256']!=sha(control/'request.json'):raise ValueError('Stage request changed')
            for retained,digest in stage['retained_hashes'].items():
                actual=sha(retained)
                if actual!=digest:
                    # Undistortion creates an auto-source config; the explicitly
                    # requested patch-config stage replaces exactly this file.
                    patch_stage=read(control/'stages/patch-config.json')
                    expected_path=str(control/'pipeline/dense/stereo/patch-match.cfg')
                    if not (path.name=='undistort.json' and retained==expected_path
                            and patch_stage['request_sha256']==sha(control/'request.json')
                            and patch_stage['retained_hashes'].get(retained)==actual):
                        raise ValueError(f'Stage checksum failed: {retained}')
                    superseded.append(dict(path=retained,old_stage='undistort',new_stage='patch-config',
                                           old_sha256=digest,current_sha256=actual))
                verified+=1
        qc=read(control/'depth_qc.json')
        if qc['maps']!=62 or qc['shape']!=[1080,1920] or not (0<qc['coverage_min']<=qc['coverage_mean']<=1):raise ValueError('Depth QC failed')
        diagnostic=root/'analysis'/frame;dr=read(diagnostic/'result.json')
        if dr['request_sha256']!=sha(diagnostic/'request.json') or dr['evidence_sha256']!=sha(diagnostic/'evidence.npz'):raise ValueError('Diagnostic changed')
        gate=root/'guarded'/frame;gr=read(gate/'result.json')
        if gr['request_sha256']!=sha(gate/'request.json') or gr['mesh_sha256']!=sha(gate/'mesh.ply'):raise ValueError('Gate changed')
        if not gr['observed_free_space_guard_passed'] or any(c['trusted_free_pixels'] for c in gr['rounds'][-1]['checks']):raise ValueError('Geometry gate failed')
        if len(gr['rounds'][-1]['checks'])!=124:raise ValueError('Expected 62 cameras and two lattices')
        if gr['depth_receipt']!=read(diagnostic/'request.json')['real_depth_receipt']:raise ValueError('Different depths for diagnosis/gate')
        parent=next(r for r in read(PARENT/'request.json')['inventory'] if r['frame_id']==frame)
        old=o3d.io.read_triangle_mesh(parent['mesh']);new=o3d.io.read_triangle_mesh(str(gate/'mesh.ply'))
        if not np.array_equal(np.asarray(old.vertices),np.asarray(new.vertices)) or not np.array_equal(np.asarray(old.triangles),np.asarray(new.triangles)[:len(old.triangles)]):raise ValueError('Original mesh changed')
        _,old_components,_=old.cluster_connected_triangles()
        labels,new_components,_=new.cluster_connected_triangles();labels=np.asarray(labels)
        islands=set(labels[len(old.triangles):])-set(labels[:len(old.triangles)])
        if islands or len(new_components)>len(old_components):raise ValueError('New disconnected triangle islands')
        def nonmanifold(mesh):
            triangles=np.asarray(mesh.triangles)
            edges=np.sort(triangles[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
            _,counts=np.unique(edges,axis=0,return_counts=True)
            return int((counts>2).sum())
        old_nm,new_nm=nonmanifold(old),nonmanifold(new)
        if new_nm>old_nm:raise ValueError('New nonmanifold edges')
        spots=read('/mnt/data/dec5_elevated_camera_jaw_review_150/end_diagnosis/spot_audit.json')['selected_components']
        x0,y0,x1,y1=next(s for s in spots if s['frame_id']==frame)['bbox_inclusive']
        crops=[];local=[];pairs=[]
        for camera in ['old_moving','phase_moving']:
            rr=[];images=[];depths=[]
            for variant in ['baseline','guarded']:
                out=root/'rgb'/frame/camera/variant;folder=out/'frames'/frame
                rc=read(folder/'complete.json')
                if rc['request_sha256']!=sha(out/'request.json'):raise ValueError('Render request changed')
                for name,digest in rc['hashes'].items():
                    if sha(folder/name)!=digest:raise ValueError('Render output changed')
                result=read(folder/'result.json');rr.append(result)
                if result['target_rgb_read'] or result['rgb_averaging'] or result['source_time_frame_count']!=1:raise ValueError('Render protocol changed')
                if len(result['source_cameras'])!=62 or 'F004_B005_1210O9' in result['source_cameras']:raise ValueError('Source split mismatch')
                image=np.asarray(Image.open(folder/'frame.png').convert('RGB'));images.append(image)
                if image.shape!=(1920,1080,3):raise ValueError('Invalid RGB shape')
                d=np.load(folder/'target_depth.npz')['depth'];depths.append(np.rot90(d))
                if not np.isfinite(d).all():raise ValueError('Nonfinite render depth')
                if camera=='old_moving':
                    crop=Image.fromarray(image).crop((x0-65,y0-65,x1+66,y1+66));crops.append(crop)
                    local.append(dict(variant=variant,depth_misses=int((depths[-1][y0:y1+1,x0:x1+1]<=0).sum()),
                        rgb_black_pixels=int((image[y0:y1+1,x0:x1+1].max(2)==0).sum())))
            if rr[0]['camera']!=rr[1]['camera'] or rr[0]['fixed_exposure']!=rr[1]['fixed_exposure']:raise ValueError('Unmatched pair')
            changed=np.any(images[0]!=images[1],axis=2)
            geometry=np.abs(depths[0]-depths[1])>1e-7
            pairs.append(dict(camera=camera,changed_rgb_pixels=int(changed.sum()),changed_depth_pixels=int(geometry.sum()),
                changed_rgb_without_depth_change=int((changed&~geometry).sum()),
                note='Localization diagnostic, not a full-frame image-quality metric'))
        panel=Image.new('RGB',(crops[0].width*2,crops[0].height+25));draw=ImageDraw.Draw(panel)
        for i,crop in enumerate(crops):panel.paste(crop,(i*crop.width,25));draw.text((i*crop.width+4,5),['baseline','guarded'][i],fill='white')
        dest=root/'rgb_review'/frame/'spot_native.png';panel.save(dest);panels.append(dict(path=str(dest),sha256=sha(dest)))
        records.append(dict(frame=frame,stage_hashes_checked=verified,depth_qc=qc,local_spot=local,pairs=pairs,
            explicitly_superseded_stage_files=superseded,
            old_components=len(old_components),new_components=len(new_components),new_triangle_islands=len(islands),
            old_nonmanifold_edges=old_nm,new_nonmanifold_edges=new_nm,
            guard_result_sha256=sha(gate/'result.json'),diagnostic_result_sha256=sha(diagnostic/'result.json')))
        train_checked=0
        for name in ['F004_E005_1210FP','M004_B005_12109O']:
            requests=[]
            for variant in ['baseline','guarded']:
                out=root/'train_rgb'/frame/name/variant;folder=out/'frames'/frame
                receipt=read(folder/'complete.json');request=read(out/'request.json');requests.append(request)
                if receipt['request_sha256']!=sha(out/'request.json'):raise ValueError('Train validation request changed')
                for key,digest in receipt['hashes'].items():
                    if sha(folder/key)!=digest:raise ValueError('Train validation output changed')
                    train_checked+=1
                if request['target_camera']!=name or not request['real_train_validation']:raise ValueError('Train validation camera mismatch')
            if requests[0]['inventory'][0]['camera']!=requests[1]['inventory'][0]['camera']:raise ValueError('Unmatched real train pair')
        records[-1]['train_render_hashes_checked']=train_checked
    if read(root/'guarded/001193/request.json')['rule']!=read(root/'guarded/001195/request.json')['rule']:raise ValueError('Per-frame rule exception')
    atomic_json(root/'audit.json',dict(script_sha256=sha(__file__),records=records,panels=panels,visual_status='requires_separate_review',
        no_full_frame_quality_metrics=True,production_replaced=False))
    print(records,flush=True)
    if (root/'visual_review.json').exists():
        visual=read(root/'visual_review.json')
        images={p:sha(root/p) for p in visual['inspected_images']}
        # All jobs must be terminal before taking this final retained-file snapshot.
        # Flush the final stdout line first because the caller retains its log.
        hashes={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
                if p.is_file() and p!=root/'artifact_manifest.json'}
        atomic_json(root/'artifact_manifest.json',dict(retained_hashes=hashes,reviewed_image_hashes=images,
            audit_sha256=sha(root/'audit.json'),visual_review_sha256=sha(root/'visual_review.json'),
            status='partial_local_improvement_not_promoted',production_replaced=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/mnt/data/dec5_jaw_measured_depth'))
    run(p.parse_args().root)
