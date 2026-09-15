"""Matched four-arm native/moving review; counts are not image-quality scores."""
from pathlib import Path
import numpy as np
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json
from study_weak_fringe_replacement import ROOT,INSET,MOVIE,FRAMES
from study_confidence_depth_prior import region_masks
from review_jaw_repair_transfer import verified_image,panel


def run():
    records=[]
    for frame in FRAMES:
        regions={k:np.rot90(v) for k,v in region_masks(frame).items()}
        yy,xx=np.nonzero(regions['hair'])
        crownbox=(max(0,int(xx.min())-15),max(0,int(yy.min())-15),min(1080,int(xx.max())+16),int(yy.min()+.25*(yy.max()-yy.min()))+35)
        for view in ['moving','native_unmasked']:
            roots=dict(production=MOVIE if view=='moving' else INSET/'rgb'/frame/view/'baseline',
                inset=INSET/'rgb'/frame/view/'completion',remove_only=ROOT/'rgb'/frame/view/'remove_only',
                replace=ROOT/'rgb'/frame/view/'replace')
            images={};depths={};receipts={};requests={}
            for arm,root in roots.items():
                images[arm],receipts[arm]=verified_image(root,frame)
                depths[arm]=np.rot90(np.load(root/'frames'/frame/'target_depth.npz')['depth'])
                requests[arm]=read(root/'request.json')
            for r in receipts.values():
                for k in ['camera','source_cameras','fixed_exposure']:assert r[k]==receipts['production'][k]
            for q in requests.values():
                for k in ['profiles_sha256','exposure_sha256','calibration_sha256']:assert q[k]==requests['production'][k]
            gt=None
            if view=='native_unmasked':
                gtpath=INSET/'review'/frame/'native_unmasked_gt.png'
                prior=next(r for r in read(INSET/'review/result.json')['records'] if r['frame']==frame and r['view']==view)
                assert sha(gtpath)==prior['gt_sha256'];gt=np.asarray(Image.open(gtpath))
            for old,new,label in [('production','remove_only','removal'),('inset','replace','replacement'),('production','replace','combined')]:
                oldvalid,newvalid=depths[old]>0,depths[new]>0
                lost=oldvalid&~newvalid;gained=newvalid&~oldvalid
                oldblack,newblack=images[old].max(2)==0,images[new].max(2)==0
                rec=dict(frame=frame,view=view,comparison=label,lost_depth_pixels=int(lost.sum()),new_depth_pixels=int(gained.sum()),
                    black_introduced=int((newblack&~oldblack).sum()),black_removed=int((oldblack&~newblack).sum()),
                    changed_rgb_pixels=int(np.any(images[old]!=images[new],2).sum()),counts_not_anatomical_metrics=True,
                    input_request_hashes={str(roots[a]/'request.json'):sha(roots[a]/'request.json') for a in [old,new]})
                if view=='native_unmasked':
                    rec['native_regions']={k:dict(lost_depth=int((lost&m).sum()),new_depth=int((gained&m).sum()),
                        black_introduced=int((newblack&~oldblack&m).sum()),black_removed=int((oldblack&~newblack&m).sum())) for k,m in regions.items()}
                records.append(rec)
                if label=='combined':continue
                imgs=[images[old],images[new]];labels=[old,new]
                if gt is not None:imgs.insert(0,gt);labels.insert(0,'real train GT')
                box=crownbox if view=='native_unmasked' else (170,450,970,850)
                panel(ROOT/'review'/frame/(view+'_'+label+'_crown.png'),imgs,labels,box)
            jawimgs=[images[k] for k in ['production','remove_only','replace']];labels=['production','remove_only','replace']
            if gt is not None:jawimgs.insert(0,gt);labels.insert(0,'real train GT')
            panel(ROOT/'review'/frame/(view+'_jaw.png'),jawimgs,labels,(400,850,900,1250))
    atomic_json(ROOT/'review/result.json',dict(records=records,script_sha256=sha(__file__),visual_status='pending',
        full_frame_quality_metrics=False,production_updated=False))
    for record in records:print(record,flush=True)


if __name__=='__main__':run()
