"""Matched RGB guard comparison; same source selection, poses and geometry prior."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json
import review_confidence_boundary_completion as scorer


def run(output,frames):
    scorer.ROOTS['chroma_guard']=Path('/mnt/data/dec5_forearm_annotation_domain_only')
    scorer.ROOTS['rgb_guard']=Path('/mnt/data/dec5_forearm_rgb_qualified_curve')
    scorer.run(output,frames,variants=['chroma_guard','rgb_guard'])
    result=read(output/'metrics.json');checks=[]
    for frame in frames:
        old=scorer.ROOTS['chroma_guard']/frame;new=scorer.ROOTS['rgb_guard']/frame
        if sha(old/'transferred.ply')!=sha(new/'transferred.ply'):raise ValueError('Changed pre-carve geometry')
        for view in ['moving','H004_A005_1210M6']:
            images=[np.array(Image.open(scorer.ROOTS[label]/'rgb'/frame/view/'guarded/frames'/frame/'frame.png')) for label in ['chroma_guard','rgb_guard']]
            checks.append(dict(frame=frame,view=view,pre_carve_geometry_exact=True,
                head_rgb_exact=bool(np.array_equal(images[0][500:1300],images[1][500:1300])),
                head_changed_pixels=int(np.any(images[0][500:1300]!=images[1][500:1300],axis=2).sum())))
            if view!='moving':
                mask=np.rot90(scorer.prior.v2.v1.masks(frame)[view])
                folders=[scorer.ROOTS[label]/'rgb'/frame/view/'guarded/frames'/frame for label in ['chroma_guard','rgb_guard']]
                depths=[np.rot90(np.load(p/'target_depth.npz')['depth']) for p in folders]
                sources=[np.rot90(np.array(Image.open(p/'source_ids.png'))) for p in folders]
                same_depth=np.isclose(depths[0],depths[1],atol=1e-7,rtol=0)
                rgb_changed=np.any(images[0]!=images[1],axis=2)
                checks[-1].update(skin_depth_changed_pixels=int((mask&~same_depth).sum()),
                    skin_rgb_changed_pixels=int((mask&rgb_changed).sum()),
                    skin_rgb_changed_at_same_depth=int((mask&rgb_changed&same_depth).sum()),
                    skin_source_changed_pixels=int((mask&(sources[0]!=sources[1])).sum()))
    result.update(wrapper_script_sha256=sha(__file__),matched_geometry_and_head_checks=checks)
    atomic_json(output/'metrics.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frames',nargs='+',default=['001029','001033','001037'])
    p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_forearm_rgb_guard_review'));a=p.parse_args();run(a.output,a.frames)
