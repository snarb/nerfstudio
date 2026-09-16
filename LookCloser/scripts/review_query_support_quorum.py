"""Actual train-view/GT comparisons and every new-black component, not metrics."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import label,find_objects
from study_multiview_face_prior import read,save,sha
from study_query_support_quorum import SOURCE,ROOT,FRAME
from review_measured_free_surface import VIEWS,baseline,DEPTH_ROOT
from review_jaw_repair_transfer import verified_image,panel


def main():
    root=ROOT/FRAME; dest=root/'review';assert not dest.exists()
    boxes={'moving':(380,1400,680,1870),'H004_C005_1210SZ':(230,1080,530,1550),'K004_B005_1210DS':(40,1120,340,1590)}
    heads={'moving':(150,900,850,1490),'H004_C005_1210SZ':(100,500,1000,1200),'K004_B005_1210DS':(100,450,950,1210)}
    records=[];bindings={str(Path(__file__)):sha(__file__)}
    for view in VIEWS:
        roots=[baseline(FRAME,view),SOURCE/'rgb'/view,root/'rgb'/view]
        images=[];depths=[];metadata=[];requests=[]
        for folder in roots:
            im,r=verified_image(folder,FRAME);images.append(im);metadata.append(r);requests.append(read(folder/'request.json'))
            depths.append(np.rot90(np.load(folder/'frames'/FRAME/'target_depth.npz')['depth']))
            for p in [folder/'request.json',folder/'frames'/FRAME/'complete.json']:bindings[str(p)]=sha(p)
        for r in metadata[1:]:
            for k in ['camera','source_cameras','fixed_exposure']:assert r[k]==metadata[0][k]
        for q in requests[1:]:
            for k in ['profiles_sha256','exposure_sha256','calibration_sha256','recipe','source_quality_implementation_sha256']:assert q[k]==requests[0][k],k
        old,new=images[1:];d0,d1=depths[1:]
        assert not ((d0<=0)&(d1>0)).any() and not ((d0>0)&(d1>0)&(d1<d0-1e-6)).any()
        names=['production','any near protects','query quorum3']
        if view!='moving':
            gt=DEPTH_ROOT/FRAME/'review'/view/'train_gt.png';images=[np.asarray(Image.open(gt)),*images];names=['real train GT',*names];bindings[str(gt)]=sha(gt)
        folder=dest/view
        panel(folder/'lipstick_native.png',images,names,boxes[view]);panel(folder/'head_native.png',images,names,heads[view])
        mask=(old.max(2)>0)&(new.max(2)==0);lab,n=label(mask);slices=find_objects(lab);components=[]
        for i,sl in enumerate(slices):
            yy,xx=sl;crop=(max(0,xx.start-15),max(0,yy.start-15),min(1080,xx.stop+15),min(1920,yy.stop+15))
            path=folder/f'new_black_{i}.png';panel(path,[old,new],['any near','quorum3'],crop)
            components.append(dict(path=str(path),pixels=int((lab==i+1).sum()),crop=crop))
        records.append(dict(view=view,changed_rgb=int(np.any(old!=new,2).sum()),newly_black_rgb=int(mask.sum()),
            lost_depth_hits=int(((d0>0)&(d1<=0)).sum()),farther_common_hits=int(((d0>0)&(d1>d0+1e-6)).sum()),new_black_components=components))
        print(view,{k:v for k,v in records[-1].items() if k!='new_black_components'},flush=True)
    e=np.load(root/'evidence.npz');d=np.load('/mnt/data/dec5_lipstick_fin_depth/000995/evidence.npz')
    ids=d['triangle_ids'];removed=e['removed_triangle_ids'];old_removed=np.load(SOURCE/'evidence.npz')['removed_triangle_ids']
    save(dest/'result.json',dict(records=records,input_hashes=bindings,
        diagnostic_triangles_before=int(np.isin(ids,old_removed).sum()),diagnostic_triangles_after=int(np.isin(ids,removed).sum()),
        representative_removed=bool(48941 in removed),diagnostic_cohort_not_used_to_select_deletion=True,
        images={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*.png')},
        counts_not_quality_metrics=True,production_accepted=False,visual_status='pending'))


if __name__=='__main__':main()
