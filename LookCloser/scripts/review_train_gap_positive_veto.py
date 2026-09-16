"""Independent positive-footprint replay and matched quorum/veto visual review."""
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import binary_fill_holes,distance_transform_edt
from study_multiview_face_prior import read,save,sha
from study_train_gap_positive_veto import ROOT,NEGATIVE,FRAME,MASKS,HAND


def audit():
    root=ROOT/FRAME;parent=NEGATIVE/FRAME;q=read(root/'request.json');r=read(root/'result.json')
    assert r['request_sha256']==sha(root/'request.json')
    assert q['positive_script_sha256']==sha(Path(__file__).with_name('study_train_gap_positive_veto.py'))
    for p,h in r['hashes'].items():assert sha(root/p)==h
    for name,key in [('request.json','parent_gap_request_sha256'),('result.json','parent_gap_result_sha256'),
                     ('evidence.npz','parent_gap_evidence_sha256'),('mask_review.json','parent_mask_review_sha256')]:
        assert sha(parent/name)==q[key]
    oldaudit=read(parent/'independent_audit.json')
    assert oldaudit['result_sha256']==q['parent_gap_result_sha256']
    assert oldaudit['request_sha256']==q['parent_gap_request_sha256']
    for p,h in q['source_masks'].items():assert sha(p)==h
    assert sha(q['mesh'])==q['mesh_sha256']
    e=np.load(root/'evidence.npz');old=np.load(parent/'evidence.npz');masks=np.load(root/'positive_masks.npz')
    points=e['points'];samples=e['sample_indices'];negative=e['negative_by_view']
    for k in ['points','sample_indices','negative_by_view']:np.testing.assert_array_equal(e[k],old[k])
    positives=[]
    for s in q['views']:
        name=s['camera'];parts=[]
        for folder,run in [(MASKS,'sam_v2'),(HAND,'sam_v1')]:
            rv=read(folder/'mask_review.json');p=folder/run/name/f'mask_{rv["selected"][name]}.png'
            assert sha(p)==q['source_masks'][str(p)];parts.append(np.array(Image.open(p))>0)
        mask=distance_transform_edt(~binary_fill_holes(parts[0]|parts[1]))<=2
        np.testing.assert_array_equal(mask,masks[name])
        c=s['camera_parameters'];pose=np.array(c['transform_matrix']);cam=(points-pose[:3,3])@pose[:3,:3];z=-cam[:,2]
        uv=np.c_[c['fl_x']*cam[:,0]/z+c['cx'],-c['fl_y']*cam[:,1]/z+c['cy']].astype(np.float32)
        xy=np.c_[uv[:,1],1919-uv[:,0]]-s['crop'][:2]
        pixel=np.floor(xy).astype(int);result=np.zeros(len(points),bool)
        for dx in [0,1]:
            for dy in [0,1]:
                x,y=pixel[:,0]+dx,pixel[:,1]+dy
                inside=(x>=0)&(x<mask.shape[1])&(y>=0)&(y<mask.shape[0])&(z>=s['depth_slab'][0])&(z<=s['depth_slab'][1])
                ii=np.flatnonzero(inside);result[ii]|=mask[y[ii],x[ii]]
        positives.append(result)
    positives=np.array(positives);np.testing.assert_array_equal(positives,e['positive_by_view'])
    denied=np.zeros(samples.shape[0],bool)
    for v in positives:
        for column in samples.T:denied|=v[column]
    base=np.zeros(samples.shape[0],bool);base[old['removed_triangle_ids']]=True
    removed=np.flatnonzero(base&~denied);np.testing.assert_array_equal(removed,e['removed_triangle_ids'])
    before=o3d.io.read_triangle_mesh(q['mesh']);after=o3d.io.read_triangle_mesh(str(root/'mesh.ply'))
    v,t=np.asarray(before.vertices),np.asarray(before.triangles)
    np.testing.assert_array_equal(points,np.concatenate([v,v[t].mean(1)]))
    np.testing.assert_array_equal(samples,np.c_[t,np.arange(len(t))+len(v)])
    np.testing.assert_array_equal(np.asarray(after.vertices),v)
    np.testing.assert_array_equal(np.asarray(after.triangles),t[~(base&~denied)])
    save(root/'independent_audit.json',dict(request_sha256=sha(root/'request.json'),result_sha256=sha(root/'result.json'),
        negative_replay_audit_sha256=sha(parent/'independent_audit.json'),positive_footprint_replay=True,
        exact_vertices_and_subset=True,removed_faces=len(removed),vetoed_faces=int((base&denied).sum()),
        physical_truth_of_masks_not_certified=True,script_sha256=sha(__file__)))


if __name__=='__main__':
    import review_train_gap_carving as review
    review.ROOT=ROOT;review.audit=audit
    review.review()
