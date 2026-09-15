"""Native before/strict/peeled/conditional review, with unchanged-region checks."""
from pathlib import Path
import numpy as np
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json
from study_measured_source_visibility import ROOT as STRICT,old_root
from render_measured_depth_peeling import ROOT as PEELED,VIEWS
from render_confidence_gated_peeling import ROOT
from review_full_block_transfer import ROOT as DEPTH_ROOT
from review_jaw_repair_transfer import verified_image,panel


def run():
    frame='000995';records=[]
    boxes={'moving':(380,1400,680,1870),'H004_C005_1210SZ':(230,1080,530,1550),
        'K004_B005_1210DS':(40,1120,340,1590)}
    heads={'moving':(150,900,850,1490),'H004_C005_1210SZ':(100,500,1000,1200),
        'K004_B005_1210DS':(100,450,950,1210)}
    for view in VIEWS:
        roots=[old_root(frame,'production',view),STRICT/frame/'production'/view,PEELED/frame/view,ROOT/frame/view]
        images=[];receipts=[]
        for root in roots:
            im,r=verified_image(root,frame);images.append(im);receipts.append(r)
        for r in receipts:
            for k in ['camera','source_cameras','mesh_sha256','fixed_exposure']:assert r[k]==receipts[0][k]
        a=images[0];b=images[-1]
        evidence=np.load(roots[-1]/'frames'/frame/'confidence_evidence.npz')
        change=np.zeros((1080,1920),bool);ids=evidence['changed_query_indices']
        change[evidence['query_y'][ids],evidence['query_x'][ids]]=True;change=np.rot90(change)
        np.testing.assert_array_equal(b[~change],a[~change])
        names=['production','strict sources','strict + deeper mesh','conditional free-space only']
        out=roots[-1]/'review';out.mkdir(exist_ok=True)
        originals=images.copy()
        if view!='moving':
            images=[np.asarray(Image.open(DEPTH_ROOT/frame/'review'/view/'train_gt.png'))]+images
            names=['real train GT']+names
        panel(out/'lipstick_native.png',images,names,boxes[view])
        panel(out/'head_native.png',[a,b],['production','conditional measured layers'],heads[view])
        panel(out/'actor.png',[np.asarray(Image.fromarray(im).resize((540,960))) for im in images],names,(0,0,540,960))
        for i,im in enumerate(originals):Image.fromarray(im).crop(boxes[view]).save(out/f'arm_{i}_lipstick_native.png')
        roi={name:(slice(box[1],box[3]),slice(box[0],box[2])) for name,box in [('head',heads[view]),('lipstick_hand',boxes[view])]}
        stats={name:dict(changed=int(np.any(a!=b,2)[s].sum()),
            black_introduced=int((a.any(2)&~b.any(2))[s].sum()),
            black_removed=int((~a.any(2)&b.any(2))[s].sum())) for name,s in roi.items()}
        records.append(dict(view=view,unchanged_outside_verified_free_points=True,
            source_requests={str(r):sha(r/'request.json') for r in roots},diagnostic_pixel_counts=stats,
            hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()},visual_status='pending'))
    atomic_json(ROOT/frame/'review_result.json',dict(records=records,counts_not_quality_metrics=True,
        production_promoted=False,mesh_improvement_claim=False))


if __name__=='__main__':run()
