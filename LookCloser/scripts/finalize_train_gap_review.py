"""Record actual visual review and mask-overlap diagnostics of both gap controls."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
from study_multiview_face_prior import read,save,sha
from study_train_gap_carving import ROOT as NEGATIVE,PARENT,FRAME,MASKS,HAND
from study_train_gap_positive_veto import ROOT as VETO
from review_measured_free_surface import VIEWS
from study_confidence_depth_prior import load_real
from review_full_block_transfer import ROOT as DEPTH_ROOT


def main():
    target=VETO/FRAME/'visual_review.json';assert not target.exists()
    q=read(NEGATIVE/FRAME/'request.json')
    rows,depths,receipt=load_real(DEPTH_ROOT,FRAME)
    assert receipt==q['depth_receipt'] and len(rows)==62
    del depths
    bindings={}
    def check(p,h):
        assert sha(p)==h,str(p)
        bindings[str(p)]=h
    for p,h in q['source_masks'].items():check(p,h)
    records=[];viewed=[]
    for p,h in q['review_images'].items():check(p,h);viewed.append(p)
    for root in [NEGATIVE,VETO]:
        base=root/FRAME;qr=read(base/'request.json');result=read(base/'result.json')
        check(base/'request.json',result['request_sha256'])
        for p,h in result['hashes'].items():check(base/p,h)
        rev=read(base/'review/result.json')
        for p,h in rev['input_hashes'].items():check(p,h)
        for p,h in rev['images'].items():check(base/'review'/p,h)
        check(base/'independent_audit.json',rev['independent_audit_sha256'])
        records.append(dict(control=root.name,removed_triangles=result['removed_triangles'],views=[]))
        for view in VIEWS:
            before=PARENT/FRAME/'rgb'/view/'frames'/FRAME/'frame.png'
            after=base/'rgb'/view/'frames'/FRAME/'frame.png'
            a=np.array(Image.open(before));b=np.array(Image.open(after))
            changed=np.any(a!=b,2);y,x=np.nonzero(changed)
            stats=dict(view=view,changed_bbox=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)])
            if view!='moving':
                s=next(s for s in q['views'] if s['camera']==view);x0,y0,x1,y1=s['crop']
                newlyblack=(a.max(2)>0)&(b.max(2)==0);crop=newlyblack[y0:y1,x0:x1]
                overlap=[];inside=[]
                for folder,run in [(MASKS,'sam_v2'),(HAND,'sam_v1')]:
                    rv=read(folder/'mask_review.json');p=folder/run/view/f'mask_{rv["selected"][view]}.png'
                    m=np.array(Image.open(p))>0
                    overlap.append(int((crop&m).sum()));inside.append(int((crop&(distance_transform_edt(m)>2)).sum()))
                stats.update(new_black_inside_tube_hand=overlap,new_black_inside_eroded_tube_hand=inside)
            if view=='K004_B005_1210DS':
                p=Image.new('1',(1080,1920));ImageDraw.Draw(p).polygon([(156,1230),(164,1230),(164,1248),(156,1248)],fill=1)
                mask=np.array(p,bool);d=np.rot90(np.load(base/'rgb'/view/'frames'/FRAME/'target_depth.npz')['depth'])
                stats.update(posthoc_core_pixels=int(mask.sum()),posthoc_core_depth_hits=int((d[mask]>0).sum()),
                    posthoc_core_colored=int((b[mask].max(1)>0).sum()))
            records[-1]['views'].append(stats)
            for name in ['lipstick_native.png','head_native.png','new_black_native_sheet.png']:
                viewed.append(str(base/'review'/view/name))
        check(base/'review/result.json',sha(base/'review/result.json'))
    assert len(viewed)==22
    for view in records[1]['views']:
        if 'new_black_inside_tube_hand' in view:assert view['new_black_inside_tube_hand']==[0,0]
    save(target,dict(reviewer='root LLM actual image inspection',actual_viewed_images={p:sha(p) for p in viewed},
        reviewed_image_count=22,checked_bindings=bindings,checked_binding_count=len(bindings),
        native_depth_receipt_reverified=True,diagnostic_records=records,
        majority_only_verdict='fail: large false bridge removed but a pink-tip notch appears in K/B',
        positive_veto_verdict='partial improvement: large bridge reduced and visible pink tip restored; lower ragged blue fringe remains',
        notes='All four mask panels and all six head/lipstick pairs and six native component sheets actually viewed. Moving view changes modestly. Head/crown defects persist; no all-frame or independent held-out quality claim.',
        mask_overlap_not_independent_quality=True,semantic_prior=True,production_promoted=False,
        artifact_free=False,video_unchanged=True,cheek_hole_goal_incomplete=True,
        script_sha256=sha(__file__)))
    save(NEGATIVE/FRAME/'visual_review.json',dict(status='fail',
        reason='Pink lipstick tip clipped in K/B; see shared review for actual inspected files',
        shared_review=str(target),shared_review_sha256=sha(target),production_promoted=False))
    print('verified',len(bindings),'bindings;22 viewed panels; foreground veto removes tip regression',flush=True)


if __name__=='__main__':main()
