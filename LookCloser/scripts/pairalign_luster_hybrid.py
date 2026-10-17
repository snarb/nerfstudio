"""Bounded TRAIN/TRAIN residual registration before view interpolation.

Only texture lookups move. Frozen geometry, calibration, grids and baseline stay
unchanged. No target photographs, evaluation masks, or learned flow model.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from luster_hybrid_common import geometry,image,log,native_rows,project,read,sha,source_path,unproject,write
from transfer_luster_hybrid import gaussian,merge_details,footprint_metrics
cv2.setNumThreads(4)


def residual_flow(a,b,support):
    h,w=support.shape
    def gray(x):
        g=cv2.cvtColor(x,cv2.COLOR_RGB2GRAY)
        return np.rint(cv2.resize(gaussian(g,1.),(max(8,w//2),max(8,h//2)),interpolation=cv2.INTER_AREA).clip(0,1)*255).astype(np.uint8)
    # Identical unsupported pixels stop the solver using unobserved texture.
    ag=gray(a);bg=gray(np.where(support[...,None],b,a))
    def solve(x,y):
        solver=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        solver.setUseMeanNormalization(True)
        f=solver.calc(x,y,None)
        return cv2.resize(f,(w,h))*np.array([w/x.shape[1],h/x.shape[0]],np.float32)
    return solve(ag,bg),solve(bg,ag)


def remap(a,uv,border=0):
    return cv2.remap(a,uv[...,0],uv[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=border)


def intermediate_lookups(forward,backward,fraction):
    yy,xx=np.indices(fraction.shape,np.float32);xy=np.stack([xx,yy],-1)
    ua=xy.copy();ub=xy.copy()
    for _ in range(3):
        ua=xy-(1-fraction[...,None])*remap(forward,ua)
        ub=xy-fraction[...,None]*remap(backward,ub)
    return ua,ub


def safe_flow(forward,backward,support,max_shift):
    yy,xx=np.indices(support.shape,np.float32);xy=np.stack([xx,yy],-1)
    error=np.linalg.norm(forward+remap(backward,xy+forward,border=100),axis=-1)
    dx=np.gradient(forward,axis=1);dy=np.gradient(forward,axis=0)
    jac=(1+dx[...,0])*(1+dy[...,1])-dx[...,1]*dy[...,0]
    safe=cv2.erode(support.astype(np.uint8),np.ones((17,17),np.uint8))>0
    safe&=(error<=.5)&(np.linalg.norm(forward,axis=-1)<=max_shift)&(jac>.5)&(jac<1.5)
    return safe


def combine(root,parent,name,max_shift=4.,full_strength=False):
    rows=native_rows();old=root/'candidates'/parent;out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    parent_config=read(old/'config.json');config={**parent_config,'pair_alignment_max_shift':max_shift,'pair_alignment_parent':parent,'full_strength':full_strength}
    write(out/'config.json',config)
    for target in [7,12,95,150]:
        if (out/f'cam{target:03d}_receipt.json').exists():continue
        g=geometry(root,target);base=np.asarray(g['rgb']);oldreceipt=read(old/f'cam{target:03d}_receipt.json')
        with np.load(old/f'cam{target:03d}_blend_provenance.npz') as b:
            x0,y0,x1,y1=map(int,b['box']);first=b['first'];second=b['second'];weight=b['first_weight'];pairs=b['pair_labels']
        sl=np.s_[y0:y1,x0:x1];shape=pairs.shape
        texture=np.asarray(base[sl]).copy();confidence=np.zeros(shape,np.float32);applied_flow=np.zeros(shape,np.uint8)
        parent_weight=np.load(old/f'cam{target:03d}_confidence.npy',mmap_mode='r')[sl]
        records={}
        for cid in oldreceipt['sources']:
            d=root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';r=read(d/'receipt.json')
            records[cid]={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']}
        def valid(a):
            return (a['valid']>0)&(a['alpha']>=config['alpha'])&(a['depth_error']<=config['depth_error'])&(a['roundtrip']<=config['roundtrip'])&(a['uncertainty']<=config['uncertainty'])
        yy,xx=np.mgrid[y0:y1,x0:x1];xy=np.stack([xx,yy],-1).astype(np.float32)
        center_world=unproject(np.asarray(g['depth_z_q50'][sl]),rows[target],xy)
        for label in np.unique(pairs[pairs>=0]):
            ca,cb=divmod(int(label),166);a=records[ca];b=records[cb];selected=pairs==label
            fraction=np.where(first==ca,weight,1-weight)
            raw=a['rgb']*fraction[...,None]+b['rgb']*(1-fraction[...,None])
            error=gaussian(np.mean(np.abs(a['rgb']-b['rgb']),axis=-1),2.)
            parent_fade=np.minimum(cv2.distanceTransform(selected.astype(np.uint8),cv2.DIST_L2,5)/3,1)*config['gain']
            confidence[selected]=1. if full_strength else (parent_weight/np.maximum(parent_fade,1e-8))[selected]
            texture[selected]=raw[selected]
            if selected.sum()<256:continue
            support=valid(a)&valid(b)
            forward,backward=residual_flow(np.asarray(a['rgb']),np.asarray(b['rgb']),support)
            ua,ub=intermediate_lookups(forward,backward,fraction)
            safea=safe_flow(forward,backward,support,max_shift);safeb=safe_flow(backward,forward,support,max_shift)
            safe=selected&(remap(safea.astype(np.float32),ua)>.999)&(remap(safeb.astype(np.float32),ub)>.999)
            # Re-read original native photos at the corrected lookup: no second RGB interpolation.
            corrected=[]
            for cid,uvlocal in [(ca,ua),(cb,ub)]:
                lookup=uvlocal+np.array([x0,y0],np.float32)
                z=remap(np.nan_to_num(np.asarray(g['depth_z_q50'])),lookup)
                world=unproject(z,rows[target],lookup);uv,_=project(world,rows[cid])
                _,expected=project(center_world,rows[cid])
                photo=np.asarray(Image.open(source_path(cid)).convert('RGB'),np.float32)/255
                rgb,ok,alpha,dep,sd,lo,hi=footprint_metrics(photo,geometry(root,cid),uv,expected,None,None,None,None,None)
                back,_=project(unproject(sd,rows[cid],uv),rows[target])
                safe&=ok&(alpha>=config['alpha'])&(dep<=config['depth_error'])&(np.linalg.norm(back-lookup,axis=-1)<=config['roundtrip'])
                corrected.append(rgb)
            after=gaussian(np.mean(np.abs(corrected[0]-corrected[1]),axis=-1),2.)
            safe&=(after+.001<error)&(after<=config['agreement'])
            moved=corrected[0]*fraction[...,None]+corrected[1]*(1-fraction[...,None])
            # Smooth transition of accepted registration within the original pair.
            fade=np.minimum(cv2.distanceTransform(safe.astype(np.uint8),cv2.DIST_L2,5)/3,1)
            texture[selected]=(raw+fade[...,None]*(moved-raw))[selected]
            applied_flow[safe]=1
            log(root,'pair_alignment',camera=target,first=ca,second=cb,pixels=int(safe.sum()))
        local,applied=merge_details(base[sl],texture,pairs,confidence,config['mode'],config['sigma'],config['gain'])
        result=base.copy();result[sl]=local
        labels=np.load(old/f'cam{target:03d}_sources.npy');labels=np.where(np.pad(applied,((y0,base.shape[0]-y1),(x0,base.shape[1]-x1)))>0,labels,-1)
        fullweight=np.zeros(base.shape[:2],np.float32);fullweight[sl]=applied
        assert np.isfinite(result).all() and np.array_equal(result[fullweight==0],base[fullweight==0])
        np.save(out/f'cam{target:03d}.npy',result);np.save(out/f'cam{target:03d}_sources.npy',labels)
        np.save(out/f'cam{target:03d}_confidence.npy',fullweight);np.save(out/f'cam{target:03d}_alignment.npy',applied_flow)
        for suffix in ['blend_provenance.npz','reasons.npy']:
            (out/f'cam{target:03d}_{suffix}').symlink_to(old/f'cam{target:03d}_{suffix}')
        image(out/f'cam{target:03d}.png',result);image(out/f'cam{target:03d}_transfer.png',np.repeat(fullweight[...,None],3,-1))
        write(out/f'cam{target:03d}_receipt.json',{**oldreceipt,'config':config,'code_sha256':sha(Path(__file__)),
              'parent_receipt_sha256':sha(old/f'cam{target:03d}_receipt.json'),'registration':'bounded classical TRAIN/TRAIN texture lookup',
              'aligned_pixels':int(applied_flow.sum()),'changed_fraction':float(np.any(result!=base,axis=-1).mean())})
        log(root,'hybrid_complete',camera=target,candidate=name,aligned_pixels=int(applied_flow.sum()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',default='bracket8_s8')
    p.add_argument('--name',default='pairalign4_bracket8_s8');p.add_argument('--max-shift',type=float,default=4.)
    p.add_argument('--full-strength',action='store_true');a=p.parse_args();combine(a.root,a.parent,a.name,a.max_shift,a.full_strength)
