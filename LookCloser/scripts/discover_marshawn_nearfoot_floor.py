"""Recover existing TRAIN feature tracks near shoe anchors; proximity is not a floor label."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import cv2


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main(a):
    sys.path.insert(0,str(a.helpers))
    from marshawn_original_feature_tracks import recover_track
    cv2.setNumThreads(1)
    r=a.scene;b=r/'tuning_20261106/native_semantic_grid_20000'
    tfpath=r/'data/transforms.json';tf=json.loads(tfpath.read_text())
    audit=json.loads((r/'geometry/audit.json').read_text())
    assert sha(tfpath)==audit['transforms_sha256'] and audit['eval_used'] is False
    rows={f['physical_camera']:f for f in tf['frames']if f['file_path']in tf['train_filenames']}
    with np.load(r/'geometry/train_witnesses.npz')as z:xyz=z['xyz'];names=z['cameras'].tolist()
    assert len(rows)==63 and set(rows)==set(names)and not set(names)&{'DC','HC','KA'}
    tracks=json.loads((r/'geometry/strict_tracks.json').read_text())['camera_ids']
    frame_path=b/'code/marshawn_1660_1900_floor.json';t=np.array(json.loads(frame_path.read_text())['world_to_box'])
    floor=xyz@t[:3,:3].T+t[:3,3]
    upper_path=b/'corrected_v2/train_shoe_feature_depth.json'
    sole_path=b/'mat_geometry_TRAIN/qualified_material_points_v2.json'
    material=json.loads(sole_path.read_text())['records']
    anchors=np.array([json.loads(upper_path.read_text())['points'][0]['xyz_floor'],next(q['xyz_floor']for q in material if q['source_point_id']==6862)])
    distance=np.linalg.norm(floor[:,None,:2]-anchors[None,:,:2],axis=2).min(1)
    ids=np.flatnonzero((distance<.2)&(floor[:,2]>-.05)&(floor[:,2]<.025))
    needed={names[i]for pid in ids for i in tracks[pid]};features={};bindings={}
    for camera in sorted(needed):
        path=r/'geometry/features'/(camera+'.npz')
        with np.load(path)as z:
            features[camera]={key:z[key]for key in ['uv','des']};source=r/'data'/rows[camera]['file_path']
            assert sha(source)==str(z['sha'])
        bindings[str(path)]=sha(path);bindings[str(source)]=sha(source)
    allowed={(p['a'],p['b'])for p in audit['pairs']};records=[];failed=[]
    prior={q['source_point_id']:q['label']for q in material}
    for pid in ids[np.argsort(distance[ids])]:
        try:g=recover_track(xyz[pid],[names[i]for i in tracks[pid]],rows,features,allowed,tolerance=1e-10)
        except ValueError as e:failed.append(dict(source_point_id=int(pid),reason=str(e)));continue
        records.append(dict(source_point_id=int(pid),xyz_floor=floor[pid].tolist(),nearest_anchor_XY_distance=float(distance[pid]),
                            verified_feature_graph=g,previous_material_label=prior.get(int(pid)),floor_qualification='UNREVIEWED'))
    for path in [tfpath,r/'geometry/audit.json',r/'geometry/strict_tracks.json',r/'geometry/train_witnesses.npz',frame_path,upper_path,sole_path,a.helpers/'marshawn_original_feature_tracks.py',a.helpers/'audit_marshawn_geometry.py']:
        bindings[str(path)]=sha(path)
    a.output.mkdir(parents=True,exist_ok=False)
    result=dict(records=records,failed=failed,source_bindings=bindings,selection='All existing9022TRAIN SfM points within.2XY of either measured shoe anchor, fitted Z in(-.05,.025). Bounds are search scope, not floor labels.',
                no_EVAL_input=True,no_model_depth=True,no_new_correspondences=True,no_plane_fit_or_carving=True,
                limitation='Original cached feature tracks with actual UV and fixed calibration; source material still needs manual qualification.')
    (a.output/'candidates.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(candidates=len(ids),recovered=len(records),failed=failed)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['scene','helpers','output']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
