"""Recheck all five render controls and inspected panels before sealing."""
from pathlib import Path
from itertools import combinations
import numpy as np
from study_multiview_face_prior import read,save,sha

ROOT=Path('/mnt/data/dec5_face_angular_visibility')
FRAMES=['001083','001119','001123','001127']


def main():
    output=ROOT/'final_manifest.json';assert not output.exists()
    notes=read(ROOT/'visual_notes.json');assert notes['production_promoted'] is False
    viewed=list((ROOT/'001123/review').glob('*.png'))
    for f in ['001083','001119','001127']:
        viewed.extend([ROOT/f/'consensus/mask_overview.png',ROOT/f/'review/overview.png',ROOT/f/'review/face.png'])
    viewed.extend((ROOT/'transfer_review').glob('*.png'));assert len(viewed)==len(set(viewed))==27
    hashes={};results=[];cameras=[];meshes=[]
    for f in FRAMES:
        for mode in (['raster','consensus'] if f=='001123' else ['consensus']):
            folder=ROOT/f/mode;q=read(folder/'request.json');r=read(folder/'result.json');a=read(folder/'independent_audit.json')
            assert q['frame']==r['frame']==f
            assert a['request_sha256']==r['request_sha256']==sha(folder/'request.json')
            assert a['result_sha256']==sha(folder/'result.json') and a['changed_points_replayed']==r['new_source_points']
            for p,h in q['input_hashes'].items():assert sha(p)==h;hashes[p]=h
            for n,h in r['hashes'].items():assert sha(folder/n)==h
            assert sha(folder/'face_masks.npz')==q['face_masks_sha256']
            assert sha(r['depth_reference'])==r['depth_sha256']
            for p,h in a['color_hashes'].items():assert sha(p)==h;hashes[p]=h
            br=read(Path(r['baseline'])/'result.json')
            if mode=='consensus':cameras.append(br['camera']);meshes.append(br['mesh_sha256'])
            results.append(dict(frame=f,mode=mode,source_changes=r['new_source_points'],rgb_changes=r['changed_rgb'],new_black=r['new_black']))
    assert len(set(meshes))==4
    centers=[np.asarray(c['transform_matrix'])[:3,3] for c in cameras]
    assert all(np.linalg.norm(a-b)>0 for a,b in combinations(centers,2))
    inputs=Path('/mnt/data/dec5_face_angular_transfer_inputs')
    for f in ['001083','001119','001127']:
        folder=inputs/f;q=read(folder/'request.json');c=read(folder/'complete.json')
        assert c['request_sha256']==sha(folder/'request.json')
        assert len(q['records'])==len(c['outputs'])==62
        assert sha(q['model']['file'])==q['model']['sha256'];hashes[q['model']['file']]=q['model']['sha256']
        for record in q['records']:
            assert sha(record['input_path'])==record['input_sha256']
            assert sha(record['source_path'])==record['source_sha256'];hashes[record['source_path']]=record['source_sha256']
        for record in c['outputs']:assert sha(record['path'])==record['sha256']
    for folder in [ROOT,inputs]:
        for p in folder.rglob('*'):
            if p.is_file():hashes[str(p)]=sha(p)
    for name in ['study_face_angular_visibility.py','review_face_angular_visibility.py','stage_face_visibility_transfer.py',
        'transfer_face_angular_visibility.py','audit_face_angular_visibility.py','diagnose_nose_face_normals.py',
        'pack_face_angular_review.py','seal_face_angular_visibility.py','audit_face_visibility_recovery.py',
        'study_face_interior_visibility.py','review_face_interior_visibility.py']:
        p=Path(__file__).with_name(name).resolve();hashes[str(p)]=sha(p)
    report=Path(__file__).resolve().parents[1]/'experiments/dec5_face_angular_visibility.md';hashes[str(report)]=sha(report)
    save(output,dict(results=results,hashes=hashes,viewed_images=[str(p) for p in viewed],
        actual_distinct_actor_times=4,actual_distinct_camera_centers=4,
        contiguous_video_reviewed=False,production_promoted=False,geometry_changed=False,
        late_nose_seam='removed on three inspected times',full_goal='not achieved'))
    for p,h in read(output)['hashes'].items():assert sha(p)==h,p
    print('sealed',len(hashes),'hash bindings and',len(viewed),'viewed panels',flush=True)


if __name__=='__main__':main()
