"""Fixed-hole replay and explicitly acknowledged visual seal for the transfer."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from study_multiview_face_prior import read,save,sha
from transfer_original_surface_backoff import panel

ROOT=Path('/mnt/data/dec5_mhr_surface_backoff_transfer')
LOCALITY=Path('/mnt/data/dec5_mhr_transfer_001195/silhouette100/locality/baseline.npz')


def prepare():
    dest=ROOT/'fixed_hole';dest.mkdir(exist_ok=False)
    r=read(ROOT/'001195/old_moving/result.json')
    folders=[Path(r['baseline']),Path(r['candidate']),ROOT/'001195/old_moving']
    xy=np.load(LOCALITY)['native_xy'];assert xy.shape==(73,2)
    images=[];counts=[]
    for name,p in zip(['production','completed mesh','same-surface backoff'],folders):
        rgb=np.array(Image.open(p/'prediction_native.png'))
        depth=np.load((p if name!='same-surface backoff' else folders[1])/'target_depth.npz')['depth']
        hit=depth[xy[:,1],xy[:,0]]>0;color=rgb[xy[:,1],xy[:,0]].max(1)>0
        counts.append(dict(variant=name,hits=int(hit.sum()),colored_hits=int((hit&color).sum()),misses=int((~hit).sum())))
        images.append((name,np.rot90(rgb)))
    portrait=np.c_[xy[:,1],1919-xy[:,0]];lo=portrait.min(0)-35;hi=portrait.max(0)+36
    box=[max(0,int(lo[0])),max(0,int(lo[1])),min(1080,int(hi[0])),min(1920,int(hi[1]))]
    panel(images,dest/'jaw_native.png',box)
    save(dest/'result.json',dict(records=counts,fixed_ray_inventory=str(LOCALITY),
        fixed_ray_inventory_sha256=sha(LOCALITY),crop=box,
        image_sha256=sha(dest/'jaw_native.png'),no_roi_used_for_recovery=True))
    print(counts,flush=True)


def seal():
    assert not (ROOT/'visual_review.json').exists()
    records=read(ROOT/'result.json')['cases'];assert len(records)==6
    bindings={};viewed={}
    for row in records:
        case=ROOT/row['frame']/row['view'];audit=read(case/'audit.json')
        assert sha(case/'audit.json')==row['audit_sha256']
        for p,h in audit['input_hashes'].items():assert sha(p)==h,p;bindings[p]=h
        for p,h in audit['images'].items():assert sha(p)==h,p;viewed[p]=h
        rr=read(case/'result.json')
        for p,h in rr['hashes'].items():assert sha(case/p)==h,p;bindings[str(case/p)]=h
        raw=Path(rr['candidate']);q=read(raw.parent.parent/'request.json')
        for n,h in q['helpers'].items():
            path=Path(__file__).with_name(n);assert sha(path)==h,path;bindings[str(path)]=h
        assert audit['statistics']['new_black_after']==0
    hole=read(ROOT/'fixed_hole/result.json')
    assert sha(LOCALITY)==hole['fixed_ray_inventory_sha256']
    assert [r['misses'] for r in hole['records']]==[73,1,1]
    assert [r['colored_hits'] for r in hole['records']]==[0,72,72]
    path=ROOT/'fixed_hole/jaw_native.png';assert sha(path)==hole['image_sha256'];viewed[str(path)]=sha(path)
    assert len(viewed)==21
    # This command is run only after main-agent inspection, not by the producer.
    for p in ROOT.rglob('*'):
        if p.is_file():bindings[str(p)]=sha(p)
    for p in [Path(__file__),Path(__file__).with_name('transfer_original_surface_backoff.py'),
              Path(__file__).parents[1]/'experiments/dec5_surface_backoff_transfer.md',
              Path(__file__).parents[1]/'tests/test_transfer_original_surface_backoff.py']:
        bindings[str(p)]=sha(p)
    save(ROOT/'visual_review.json',dict(status='reviewed_partial_improvement_not_production_acceptance',
        reviewed_images=viewed,checked_bindings=bindings,
        notes=['Two-time / six-view check; all 9 new black original-surface pixels recover.',
               'Known 001195 old-moving jaw puncture remains 73 to 1 misses, with 72 colored fills.',
               'Six uncolored-component panels show seven new geometric hits at clothing margin, still black.',
               'Ragged hair, neck and shoulder outlines remain. No temporal or artifact-free approval.',
               'Saved mesh is unchanged by this texture policy; no 6K video replaced.'],
        actual_main_agent_visual_review=True,full_goal_complete=False))
    print('sealed',len(bindings),'bindings;',len(viewed),'actually viewed images',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','seal'])
    p.add_argument('--confirm-reviewed',action='store_true');a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:
        assert a.confirm_reviewed,'Inspect all 21 panels before sealing'
        seal()
