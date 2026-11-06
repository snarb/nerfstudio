"""Native mask-ray coverage and source-SIFT anchor consistency of the common box."""
import argparse
import json
from pathlib import Path

import numpy as np
from mask_visual_hull_aabb import ray_box_hits
from aabb_visualization import camera_in_box_frame
from prepare_marshawn_color_variants import sha

ROOT=Path('/fsx/tmp/marshawn_001660_hd')


def rays(c,xx,yy,T,bounds):
    cc=camera_in_box_frame(c,T);pose=np.asarray(cc['transform_matrix'])
    d=np.c_[(xx+.5-c['cx'])/c['fl_x'],-(yy+.5-c['cy'])/c['fl_y'],-np.ones(len(xx))]@pose[:3,:3].T
    return ray_box_hits(pose[:3,3],d,bounds)


def main(a):
    box=json.loads((a.geometry/'common_box.json').read_text());T=np.array(box['world_to_box']);bounds=np.array(box['aabb'])
    ids=json.loads((a.input/'reviewed_identities.json').read_text())
    report=[];sfm=[]
    old=json.loads((ROOT/'two_actor_isolation/automatic_aabb/input_v2/manifest.json').read_text())
    for f in box['sample_frames']:
        masks=[]
        if f==1660:
            for r in old['cameras']:
                p=ROOT/'two_actor_isolation/automatic_aabb/input_v2'/r['masks']
                z=np.load(p);fg=z['foreground'];un=z['unknown'];c=r['camera_row']
                yy,xx=np.where(fg);hits=rays(c,xx,yy,T,bounds)
                report.append(dict(frame=f,camera=r['camera'],scope='all66 HD footprints of accepted native core + hair',
                    pixels=len(xx),misses=int((~hits).sum())))
                masks.append((c,fg,un))
        else:
            for cam,groups in ids[str(f)].items():
                p=a.input/'candidates'/str(f)/cam/'candidates.npz'
                with np.load(p) as z:target=z['masks'][sorted(set(sum(groups.values(),[])))].any(0)
                r=json.loads((a.input/'sources'/str(f)/cam/'receipt.json').read_text());c=r['native_camera']
                yy,xx=np.where(target);misses=0;examples=[]
                for start in range(0,len(xx),200000):
                    x,y=xx[start:start+200000],yy[start:start+200000]
                    hit=rays(c,x,y,T,bounds);misses+=int((~hit).sum())
                    examples+=np.c_[x[~hit],y[~hit]][:max(0,16-len(examples))].tolist()
                report.append(dict(frame=f,camera=cam,scope='every selected native undistorted crop pixel',
                    pixels=len(xx),misses=misses,miss_examples=examples,candidates_sha256=sha(p)))
                z=np.load(a.geometry/str(f)/(cam+'_mask.npz'))
                masks.append((r['camera_row'],z['foreground'],z['unknown']))
        if f==1660:
            p=ROOT/'two_actor_isolation/actor_sfm_qualification_FULLSCENE/actor_FULLSCENE_SEARCH_ONLY.npz'
            xyz=np.load(p)['witness_xyz']
            selected=xyz@T[:3,:3].T+T[:3,3]
        else:
            p=a.geometry/str(f)/'actor_anchors.npz'
            selected=np.load(p)['box_xyz']
        inside=((selected>=bounds[0])&(selected<=bounds[1])).all(-1)
        sfm.append(dict(frame=f,qualified_actor_witnesses=len(selected),outside=int((~inside).sum()),
            source_sha256=sha(p),source=str(p),
            minimum=selected.min(0).tolist() if len(selected) else None,
            maximum=selected.max(0).tolist() if len(selected) else None,
            scope='Synchronized source SIFT tracks; later-frame anchors also participate in box construction, so containment is a consistency check, not held-out validation'))
        print(f,'ray_misses',sum(r['misses'] for r in report if r['frame']==f),'SfM',sfm[-1],flush=True)
    result=dict(box_sha256=sha(a.geometry/'common_box.json'),records=report,SfM=sfm,
        total_pixels=sum(r['pixels'] for r in report),total_ray_misses=sum(r['misses'] for r in report),
        total_SfM_outside=sum(r['outside'] for r in sfm),
        limits='Mask-ray intersection is necessary, not proof of surface containment; sparse SfM does not cover every hair/limb. Five sampled times only.',
        no_training=True)
    (a.geometry/'coverage.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True)
    main(p.parse_args())
