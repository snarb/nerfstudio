"""View-bracketing two-source appearance interpolation ablation."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from luster_hybrid_common import (assert_source,geometry,image,log,native_rows,read,sha,source_path,unproject,write)
from transfer_luster_hybrid import DEFAULT,gaussian,merge_details
CONFIG={**DEFAULT,'count':8,'mode':'detail','sigma':8.,'full_strength':False}


def bracket_weight(first,second,target):
    delta=first-second
    fraction=(np.sum((target-second)*delta,axis=-1)/np.maximum(np.sum(delta*delta,axis=-1),1e-10)).clip(0,1)
    interpolated=first*fraction[...,None]+second*(1-fraction[...,None])
    error=np.linalg.norm(interpolated-target,axis=-1)
    return fraction,error


def combine(root,target,name,config):
    rows=native_rows();base=np.asarray(geometry(root,target)['rgb'])
    source_ids=read(root/'preflight/sources.json')['sources'][str(target)][:config['count']]
    out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    config_file=out/'config.json'
    if config_file.exists():assert read(config_file)==config
    else:write(config_file,config)
    directory=root/'warps'/f'cam{target:03d}'
    receipt=read(directory/f'src{source_ids[0]:03d}'/'receipt.json');x0,y0,x1,y1=receipt['box']
    shape=(y1-y0,x1-x0);sl=np.s_[y0:y1,x0:x1]
    records=[]
    for cid in source_ids:
        assert_source(rows,target,cid)
        d=directory/f'src{cid:03d}';r=read(d/'receipt.json');assert r['box']==receipt['box']
        records.append((cid,{k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']}))
    # Three best geometric candidates; a second independent source must confirm local appearance.
    scores=np.full((4,*shape),np.inf,np.float32);ids=np.full((4,*shape),-1,np.int16)
    colors=np.zeros((4,*shape,3),np.float32)
    reasons=np.zeros(shape,np.uint8);support_count=np.zeros(shape,np.uint8)
    for cid,a in records:
        valid=(a['valid']>0)&(a['alpha']>=config['alpha'])&(a['depth_error']<=config['depth_error'])
        valid&=(a['roundtrip']<=config['roundtrip'])&(a['uncertainty']<=config['uncertainty'])
        valid&=(a['score']<=config['max_score'])&(a['resolution_ratio']>=config['min_resolution'])
        support_count+=valid
        score=np.where(valid,a['score']+.1*a['roundtrip']/max(config['roundtrip'],1e-8),np.inf)
        color=np.asarray(a['rgb']);label=np.full(shape,cid,np.int16)
        for rank in range(4):
            take=score<scores[rank]
            oldscore=scores[rank].copy();oldid=ids[rank].copy();oldcolor=colors[rank].copy()
            scores[rank][take]=score[take];ids[rank][take]=label[take];colors[rank][take]=color[take]
            score=np.where(take,oldscore,score);label=np.where(take,oldid,label);color=np.where(take[...,None],oldcolor,color)
    # Bracket the target viewing direction rather than always taking the nearest
    # single camera. This is interpolation of observed appearance, not new geometry.
    g=geometry(root,target);yy,xx=np.mgrid[y0:y1,x0:x1]
    world=unproject(np.asarray(g['depth_z_q50'][sl]),rows[target],np.stack([xx,yy],-1).astype(np.float32))
    positions=np.zeros((166,3),np.float32)
    for cid,row in rows.items():positions[cid]=np.asarray(row['transform_matrix'])[:3,3]
    direction=positions[target]-world;direction/=np.maximum(np.linalg.norm(direction,axis=-1,keepdims=True),1e-8)
    views=[]
    for rank in range(4):
        vector=positions[np.maximum(ids[rank],0)]-world
        views.append(vector/np.maximum(np.linalg.norm(vector,axis=-1,keepdims=True),1e-8))
    chosen=np.full(shape,-1,np.int16);texture=np.asarray(base[sl]).copy();confidence=np.zeros(shape,np.float32)
    pair_labels=np.full(shape,-1,np.int16);first=np.full(shape,-1,np.int16);second=first.copy()
    source_weight=np.zeros(shape,np.float32);best=np.full(shape,np.inf,np.float32)
    for i in range(4):
        for j in range(i+1,4):
            pair=(ids[i]>=0)&(ids[j]>=0)
            mass=gaussian(pair.astype(np.float32),2.)
            error=gaussian(np.mean(np.abs(colors[i]-colors[j]),axis=-1)*pair,2.)/np.maximum(mass,1e-8)
            fraction,view_error=bracket_weight(views[i],views[j],direction)
            cost=view_error+.01*(scores[i]+scores[j])
            agree=pair&(mass>.8)&(error<=config['agreement'])&(cost<best)
            blended=colors[i]*fraction[...,None]+colors[j]*(1-fraction[...,None])
            texture[agree]=blended[agree]
            chosen[agree]=np.where(fraction>=.5,ids[i],ids[j])[agree]
            first[agree]=ids[i][agree];second[agree]=ids[j][agree];source_weight[agree]=fraction[agree]
            pair_labels[agree]=(np.minimum(ids[i],ids[j])*166+np.maximum(ids[i],ids[j]))[agree]
            confidence[agree]=1. if config['full_strength'] else np.clip(1-error[agree]/(2*config['agreement']),.5,1)
            best[agree]=cost[agree]
    # Remove isolated labels and shrink unsupported islands; never create new support.
    for cid in source_ids:
        m=(chosen==cid).astype(np.uint8)
        coherent=cv2.medianBlur(m,3)>0
        reject=(m>0)&~coherent;chosen[reject]=-1;confidence[reject]=0
    reasons[support_count==0]=1;reasons[support_count==1]=2;reasons[(support_count>=2)&(chosen<0)]=3
    pair_labels[chosen<0]=-1
    local,applied=merge_details(np.asarray(base[sl]),texture,pair_labels,confidence,config['mode'],config['sigma'],config['gain'])
    result=base.copy();result[sl]=local
    full_labels=np.full(base.shape[:2],-1,np.int16);full_labels[sl]=np.where(applied>0,chosen,-1)
    full_weight=np.zeros(base.shape[:2],np.float32);full_weight[sl]=applied
    full_reasons=np.full(base.shape[:2],4,np.uint8);full_reasons[sl]=reasons
    assert np.array_equal(result[full_weight==0],base[full_weight==0])
    np.savez_compressed(out/f'cam{target:03d}_blend_provenance.npz',box=np.array([x0,y0,x1,y1]),
                        first=first,second=second,first_weight=source_weight,pair_labels=pair_labels)
    np.save(out/f'cam{target:03d}.npy',result)
    np.save(out/f'cam{target:03d}_sources.npy',full_labels)
    np.save(out/f'cam{target:03d}_confidence.npy',full_weight)
    np.save(out/f'cam{target:03d}_reasons.npy',full_reasons)
    image(out/f'cam{target:03d}.png',result)
    image(out/f'cam{target:03d}_transfer.png',np.repeat(full_weight[...,None],3,-1))
    palette=np.random.default_rng(42).uniform(.15,1,(166,3)).astype(np.float32)
    color_map=palette[np.maximum(full_labels,0)];color_map[full_labels<0]=0
    image(out/f'cam{target:03d}_sources.png',color_map)
    write(out/f'cam{target:03d}_receipt.json',dict(config=config,camera=target,sources=source_ids,
          source_hashes={str(source_path(c)):sha(source_path(c)) for c in source_ids},
          code_sha256=sha(Path(__file__)),uses_target_rgb=False,uses_masks=False,
          changed_fraction=float(np.any(result!=base,axis=-1).mean()),
          reasons={'1':'no geometric source','2':'one source only','3':'appearance/coherence rejected','4':'outside predicted support'}))
    log(root,'hybrid_complete',camera=target,candidate=name,changed_fraction=float(np.any(result!=base,axis=-1).mean()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--config',type=Path)
    p.add_argument('--cameras',type=int,nargs='+',default=[7,12,95,150]);a=p.parse_args()
    config={**CONFIG,**(read(a.config) if a.config else {})}
    for c in a.cameras:combine(a.root,c,a.name,config)
