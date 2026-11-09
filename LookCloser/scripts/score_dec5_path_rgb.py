#!/usr/bin/env python3
"""Photometric continuity counterfactual on exactly frozen path geometry.

The common domain/flow comes from the dynamic-source reference. Geometry is
asserted identical in both arms. This is estimated-motion RGB continuity, not
novel-view RGB ground truth or a claim that real specular changes are errors.
"""
from pathlib import Path
import argparse,json
import cv2
import numpy as np
from dec5_hair import project,unproject
from review_dec5_hair_path import remap
from run_dec5_hair_campaign import atomic_json,sha


def score(path,dynamic,fixed,output):
    if output.exists():raise FileExistsError('RGB continuity metrics are immutable')
    cv2.setNumThreads(2)
    frames=json.loads(path.read_text())['frames']
    receipts=[json.loads((p/'complete.json').read_text()) for p in (dynamic,fixed)]
    if any(r.get('uses_validation_rgb') or r.get('uses_test_rgb') for r in receipts):raise ValueError('Only train-rendered virtual paths are accepted')
    pools={label:{roi:[] for roi in ['hair','interior','boundary']} for label in ['dynamic','fixed']};rows=[]
    previous=None
    for i,camera in enumerate(frames):
        name=camera['physical_camera'];arrays=[np.load(p/(name+'.npz')) for p in (dynamic,fixed)]
        for key in ['front_depth','back_depth','projected_hair_depth','roi']:
            if not np.array_equal(arrays[0][key],arrays[1][key]):raise ValueError('Counterfactual geometry/domain changed: '+key)
        images=[cv2.imread(str(p/(name+'.png'))).astype(np.float32)/255 for p in (dynamic,fixed)]
        if previous is not None:
            old,old_images=previous;domain=old['roi']&(old['hair_probability']>.5)&(old['source_id']>=0)
            interior=cv2.erode(domain.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
            yy,xx=np.nonzero(domain);depth=old['front_depth'][yy,xx]
            xyz=unproject(np.stack([xx+.5,yy+.5],-1),depth,frames[i-1]);uv,z=project(xyz,camera)
            following_depth=remap(arrays[0]['front_depth'],uv,True)
            tolerance=np.maximum(2*z/min(camera['fl_x'],camera['fl_y']),.002)
            valid=((depth>0)&(z>0)&(uv[:,0]>=.5)&(uv[:,0]<=camera['w']-.5)&(uv[:,1]>=.5)&(uv[:,1]<=camera['h']-.5)
                   &~((following_depth>0)&(following_depth<z-tolerance))
                   &(remap(arrays[0]['source_id'],uv,True)>=0)&(remap(arrays[0]['hair_probability'],uv,True)>.5))
            types={'hair':valid,'interior':valid&interior[yy,xx],'boundary':valid&~interior[yy,xx]}
            row=dict(pair=[i-1,i],pixels={k:int(v.sum()) for k,v in types.items()})
            for label,old_image,image in zip(['dynamic','fixed'],old_images,images):
                error=np.abs(old_image[yy,xx]-remap(image,uv)).mean(-1)
                row[label]={}
                for roi,mask in types.items():
                    values=error[mask];pools[label][roi].append(values)
                    row[label][roi]=dict(mae=float(values.mean()) if len(values) else None)
            rows.append(row)
        previous=(arrays[0],images)
    aggregate={}
    for label,types in pools.items():
        aggregate[label]={}
        for roi,values in types.items():
            values=np.concatenate(values)
            aggregate[label][roi]=dict(pixels=len(values),mae=float(values.mean()),median=float(np.median(values)),p95=float(np.quantile(values,.95)))
    atomic_json(output,dict(aggregate=aggregate,per_pair=rows,frame_count=len(frames),
        rgb='final display RGB in [0,1], all channels equally weighted',
        domain='fixed dynamic-source visible hair; train-projected semantic proxy, no GT alpha',
        motion='same frozen receiver geometry in both arms, common occlusion veto',
        all_geometry_and_roi_exact=True,uses_validation_rgb=False,uses_test_rgb=False,
        metric_kind='estimated-motion photometric continuity; not novel-view GT fidelity',
        dynamic_receipt_sha256=sha(dynamic/'complete.json'),fixed_receipt_sha256=sha(fixed/'complete.json'),
        path_sha256=sha(path),implementation_sha256=sha(__file__)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['path','dynamic','fixed','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();score(a.path,a.dynamic,a.fixed,a.output)


if __name__=='__main__':main()
