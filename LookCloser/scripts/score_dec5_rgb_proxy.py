#!/usr/bin/env python3
"""RGB continuity on common baseline-depth correspondences, including dropouts.

Unlike score_dec5_path_rgb, candidates may change geometry. This is a proxy;
deleting hair can lower variation, so coverage/visibility metrics are required.
"""
from pathlib import Path
import argparse,json
import cv2,numpy as np
from dec5_hair import unproject,project
from review_dec5_hair_path import remap
from run_dec5_hair_campaign import atomic_json,sha


def main():
    p=argparse.ArgumentParser()
    for key in ['path','baseline','candidate','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();frames=json.loads(a.path.read_text())['frames'];previous=None;totals={}
    for frame in frames:
        name=frame['physical_camera'];arrays=[dict(np.load(d/(name+'.npz'))) for d in [a.baseline,a.candidate]]
        np.testing.assert_array_equal(arrays[0]['roi'],arrays[1]['roi'])
        images=[cv2.imread(str(d/(name+'.png'))).astype(np.float32)/255 for d in [a.baseline,a.candidate]]
        if previous:
            old,old_images,camera=previous;base=arrays[0]
            domain=old['roi']&(old['hair_probability']>.5)&(old['source_id']>=0)&(old['front_depth']>0)
            y,x=np.nonzero(domain);depth=old['front_depth'][y,x]
            uv,z=project(unproject(np.stack([x+.5,y+.5],-1),depth,camera),frame)
            following=remap(base['front_depth'],uv,True);tolerance=np.maximum(2*z/min(frame['fl_x'],frame['fl_y']),.002)
            valid=(z>0)&(uv[:,0]>=.5)&(uv[:,0]<=frame['w']-.5)&(uv[:,1]>=.5)&(uv[:,1]<=frame['h']-.5)&~((following>0)&(following<z-tolerance))
            missing=remap(base['source_id'],uv,True)<0
            for region,mask in [('all_anchored_hair',valid),('baseline_fallback_transitions',valid&missing)]:
                if region not in totals:totals[region]=dict(pixels=0,baseline_abs_rgb_sum=0.,candidate_abs_rgb_sum=0.)
                totals[region]['pixels']+=int(mask.sum())
                for label,old_image,image in zip(['baseline','candidate'],old_images,images):
                    error=np.abs(old_image[y,x]-remap(image,uv)).mean(-1)
                    totals[region][label+'_abs_rgb_sum']+=float(error[mask].sum())
        previous=(arrays[0],images,frame)
    for value in totals.values():
        for label in ['baseline','candidate']:value[label+'_mae']=value[label+'_abs_rgb_sum']/max(value['pixels'],1)
    atomic_json(a.output,dict(regions=totals,baseline_sha256=sha(a.baseline/'complete.json'),
        candidate_sha256=sha(a.candidate/'complete.json'),metric='common baseline-depth transported RGB; dropout-inclusive proxy',
        deletion_can_reduce_variation=True,coverage_review_required=True,no_geometry_ground_truth=True))


if __name__=='__main__':main()
