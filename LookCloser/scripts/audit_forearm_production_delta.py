"""Compare plane/quadric production canaries without declaring visual success."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json
from PIL import Image
import study_forearm_plane_transfer_v3 as prior
from study_confidence_depth_prior import project_integer

ROOTS=dict(plane=Path('/mnt/data/dec5_forearm_production_delta'),quadratic=Path('/mnt/data/dec5_forearm_production_curved_referenced'))
PARENT=Path('/mnt/data/dec5_phase30_dynamic_150')


def run(output):
    output.mkdir(parents=True,exist_ok=True);prior.configure();v1=prior.v2.v1
    parent=read(PARENT/'request.json');records=[];metrics=[]
    for frame in ['001029','001033','001037']:
        source=next(r for r in parent['inventory'] if r['frame_id']==frame)
        if sha(source['mesh'])!=source['mesh_sha256']:raise ValueError('Changed production input')
        old=o3d.io.read_triangle_mesh(source['mesh']);ov,ot=np.asarray(old.vertices),np.asarray(old.triangles)
        _,oldcc,_=old.cluster_connected_triangles();comparisons={}
        for name,root in ROOTS.items():
            folder=root/frame;r=read(folder/'geometry_result.json')
            if r['request_sha256']!=sha(folder/'request.json'):raise ValueError('Changed geometry request')
            for p,h in r['hashes'].items():
                if sha(folder/p)!=h:raise ValueError('Changed geometry artifact')
            for p,h in r['depth_hashes'].items():
                if sha(p)!=h:raise ValueError('Changed observed depth')
            if not r['observed_guard_passed'] or len(r['rounds'][-1]['checks'])!=124 or any(c['trusted_free_pixels'] for c in r['rounds'][-1]['checks']):raise ValueError('Measured-depth guard failed')
            mesh=o3d.io.read_triangle_mesh(str(folder/'guarded.ply'));v,t=np.asarray(mesh.vertices),np.asarray(mesh.triangles)
            if not np.array_equal(v[:len(ov)],ov) or not np.array_equal(t[:len(ot)],ot):raise ValueError('Changed production prefix')
            _,cc,_=mesh.cluster_connected_triangles()
            record=dict(frame=frame,model=name,added_triangles=r['final_added_triangles'],old_components=len(oldcc),new_components=len(cc),
                        geometry_result_sha256=sha(folder/'geometry_result.json'),views=[],
                        guard_kind=read(folder/'request.json')['observed_guard'].get('kind','depth_only'))
            if r.get('color_guard_provenance'):
                fresh=read(root/'fresh_audit'/(frame+'.json'))
                if fresh['geometry_result_sha256']!=sha(folder/'geometry_result.json') or fresh['qualified_veto_pixels']!=0:
                    raise ValueError('Missing fresh color-qualified ray audit')
                record['depth_only_final_veto_pixels']=fresh['original_depth_only_veto_pixels']
                for p,h in r['color_guard_provenance']['source_rgb_hashes'].items():
                    if sha(p)!=h:raise ValueError('Changed color witness RGB')
            for view in ['moving',v1.NAMES[1]]:
                images=[];depths=[];results=[];ids=[]
                for variant in ['baseline','guarded']:
                    dest=root/'rgb'/frame/view/variant;d=dest/'frames'/frame;receipt=read(d/'complete.json')
                    if receipt['request_sha256']!=sha(dest/'request.json'):raise ValueError('Changed render request')
                    for p,h in receipt['hashes'].items():
                        if sha(d/p)!=h:raise ValueError('Changed render result')
                    result=read(d/'result.json');results.append(result)
                    if len(result['source_cameras'])!=62 or result['rgb_averaging'] or result['target_rgb_read']:raise ValueError('Changed renderer protocol')
                    im=np.asarray(Image.open(d/'frame.png').convert('RGB'));dep=np.rot90(np.load(d/'target_depth.npz')['depth'])
                    if im.shape!=(1920,1080,3) or not np.isfinite(dep).all():raise ValueError('Invalid native render')
                    images.append(im);depths.append(dep);ids.append(np.rot90(np.asarray(Image.open(d/'source_ids.png'))))
                if results[0]['camera']!=results[1]['camera'] or results[0]['fixed_exposure']!=results[1]['fixed_exposure']:raise ValueError('Unmatched pair')
                added=(depths[1]>0)&(depths[0]==0);row=dict(view=view,new_geometry_pixels=int(added.sum()),
                    new_geometry_without_rgb=int((added&(images[1].max(2)==0)).sum()),
                    changed_rgb_upper_1000_rows=int(np.any(images[0][:1000]!=images[1][:1000],axis=2).sum()))
                if view==v1.NAMES[1]:
                    mask=np.rot90(v1.masks(frame)[view]);own=results[0]['source_cameras'].index(view)
                    row['fixed_skin_geometry_missing']=[int((mask&(d==0)).sum()) for d in depths]
                    row['fixed_skin_rgb_missing']=[int((mask&(im.max(2)==0)).sum()) for im in images]
                    row['own_camera_skin_pixels']=[int((mask&(s==own)).sum()) for s in ids];row['skin_pixels']=int(mask.sum())
                    row['source_histograms']=[]
                    for s in ids:
                        labels,counts=np.unique(s[mask],return_counts=True)
                        row['source_histograms'].append({results[0]['source_cameras'][int(k)] if k<62 else 'missing':int(n) for k,n in zip(labels,counts)})
                record['views'].append(row);comparisons[name,view]=images[0]
            metrics.extend([dict(model=name,**m) for m in read(root/'metrics.json')['rows'] if m['frame']==frame])
            records.append(record)
        for view in ['moving',v1.NAMES[1]]:
            for name in ROOTS:
                if not np.array_equal(comparisons['plane',view],comparisons[name,view]):raise ValueError('Baseline replay changed')
        # Diagnose the appended old-depth boundary ring, distinct from source mesh vertices.
        base=o3d.io.read_triangle_mesh(read(prior.OUT/frame/'input.json')['mesh'])
        raw=o3d.io.read_triangle_mesh(str(prior.OUT/frame/'plane/mesh.ply'))
        rows,_,_=v1.cameras(frame);reference=next(c for c in rows if c['physical_camera']==v1.NAMES[0])
        used=np.unique(np.asarray(raw.triangles)[len(base.triangles):]);used=used[used>=len(base.vertices)]
        uv,_=project_integer(reference,np.asarray(raw.vertices)[used]);xy=np.rint(uv).astype(int)
        accepted=np.load(prior.OUT/frame/'plane/evidence.npz')['accepted'];ring=~accepted[xy[:,1],xy[:,0]]
        for record in records:
            if record['frame']!=frame or record['model']=='plane':continue
            policy=read(ROOTS[record['model']]/frame/'request.json')['curvature_policy']
            record['appended_active_old_depth_ring_vertices']=int(ring.sum())
            record['curvature_ring_handling']=('Exact old-depth ring; smooth interior feather' if policy.get('boundary_ring_exact') else
                'Moved along with other referenced appended vertices; source mesh prefix unchanged. Boundary-condition limitation.')
    atomic_json(output/'result.json',dict(script_sha256=sha(__file__),records=records,metrics=metrics,
        metric_scope='fixed real-train forearm skin only, not heldout or full-frame',production_video_unchanged=True,
        baseline_replay_bytes_equal=True,visual_status='requires_explicit_review'))
    for record in records:
        print(record['frame'],record['model'],'added_triangles',record['added_triangles'],
              'components',record['old_components'],'->',record['new_components'],flush=True)
    if (output/'visual_review.json').exists():
        review=read(output/'visual_review.json')
        expected={(r['frame'],r['model']) for r in records}
        verdicts=review['verdicts']
        if len(verdicts)!=len(expected) or {(r['frame'],r['model']) for r in verdicts}!=expected:
            raise ValueError('Incomplete or duplicated visual inventory')
        if any(r['status'] not in ['pass','fail','uncertain'] for r in verdicts):
            raise ValueError('Invalid visual verdict')
        atomic_json(output/'artifact_manifest.json',dict(reviewed_images={p:sha(p) for p in review['inspected_images']},
            audit_sha256=sha(output/'result.json'),visual_review_sha256=sha(output/'visual_review.json'),
            retained_hashes={str(p):sha(p) for root in [*ROOTS.values(),Path('/mnt/data/dec5_forearm_multiview_anchors')]
                             for p in sorted(root.rglob('*')) if p.is_file() and not p.is_relative_to(output)},status='remaining_artifacts_not_promoted'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_forearm_production_review'))
    p.add_argument('--boundary-root',type=Path);p.add_argument('--color-root',type=Path);a=p.parse_args()
    if a.boundary_root:ROOTS['boundary_quadratic']=a.boundary_root
    if a.color_root:ROOTS['color_qualified_quadratic']=a.color_root
    run(a.output)
