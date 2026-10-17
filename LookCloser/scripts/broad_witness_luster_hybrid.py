"""Extend selected-photo detail only with positive TRAIN restoration evidence.

All output RGB still comes from the frozen target and one selected native photo.
Other photos evaluate a scalar restoration utility for the additional G6-G16
band. No texture-coordinate changes, target GT or evaluation masks are used.
"""
from pathlib import Path
import cv2
import numpy as np
from luster_hybrid_common import assert_source,geometry,image,log,native_rows,read,sha,unproject,write
from supported_luster_hybrid import geometric_support
from transfer_luster_hybrid import gaussian
from residual_luster_hybrid import model_warp
from witness_benefit_luster_hybrid import tangent_cosine
from band_witness_luster_hybrid import assemble
cv2.setNumThreads(4)


def split_bands(rgb,support):
    sf=support.astype(np.float32)
    def low(sigma):
        mass=gaussian(sf,sigma)
        return gaussian(rgb*sf[...,None],sigma)/np.maximum(mass[...,None],1e-8),mass
    six,_=low(6.);sixteen,mass=low(16.)
    return rgb-six,six-sixteen,mass


def restoration_utility(delta,witness_residual,support):
    """Improvement of a candidate correction over zero correction, RGB MSE."""
    mass=gaussian(support.astype(np.float32),6.)
    def mse(value):
        error=np.mean(value*value,axis=-1)
        return gaussian(np.where(support,error,0.),6.)/np.maximum(mass,1e-8)
    before=mse(witness_residual);after=mse(delta-witness_residual)
    utility=(before-after)/(before+after+.002**2)
    return utility,support&(mass>=.8)&np.isfinite(utility)


def witness_median(utilities,known):
    """Median of all usable witnesses, with two distinct observations required."""
    count=known.sum(axis=0)
    # Sort with invalid observations at the end; avoid all-NaN median warnings.
    values=np.sort(np.where(known,utilities,np.inf),axis=0)
    lo=np.maximum((count-1)//2,0);hi=np.maximum(count//2,0)
    first=np.take_along_axis(values,lo[None],axis=0)[0]
    second=np.take_along_axis(values,hi[None],axis=0)[0]
    return np.where(count>=2,(first+second)*.5,0).astype(np.float32),count


def positive_gate(utility,count,support,threshold):
    assert 0<=threshold<1
    gate=np.clip((utility-threshold)/(1-threshold),0,1)
    gate=np.where((count>=2)&support,gate,0).astype(np.float32)
    # Positive evidence cannot spread into an unsupported/contradictory neighbor.
    return cv2.erode(gate,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)


def evidence(root,target,parent):
    old=root/'candidates'/parent;pr=old/f'cam{target:03d}_receipt.json';receipt=read(pr)
    config=receipt['config'];rows=native_rows();warp_root=Path(receipt['warp_root'])
    out=root/'broad_witness'/parent/f'cam{target:03d}';out.mkdir(parents=True,exist_ok=True)
    if (out/'receipt.json').exists():
        r=read(out/'receipt.json');assert r['code_sha256']==sha(Path(__file__))
        assert all(sha(Path(k))==v for k,v in r['inputs'].items())
        assert all(sha(out/k)==v for k,v in r['hashes'].items());return out
    base=geometry(root,target)['rgb'];deltas=[];residuals=[];valid=[]
    inputs={str(pr):sha(pr)}
    for module in ['supported_luster_hybrid.py','transfer_luster_hybrid.py','residual_luster_hybrid.py',
                   'witness_benefit_luster_hybrid.py','luster_hybrid_common.py']:
        p=Path(__file__).with_name(module);inputs[str(p)]=sha(p)
    for cid in receipt['sources']:
        assert_source(rows,target,cid)
        d=warp_root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';r=read(d/'receipt.json')
        inputs[str(d/'receipt.json')]=sha(d/'receipt.json')
        a={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']}
        support=geometric_support(a,config);x0,y0,x1,y1=r['box'];sl=np.s_[y0:y1,x0:x1]
        _,photo_band,mass=split_bands(np.asarray(a['rgb']),support)
        _,target_band,_=split_bands(np.asarray(base[sl]),support)
        _,model_band,_=split_bands(model_warp(root,target,cid),support)
        deltas.append(photo_band-target_band);residuals.append(photo_band-model_band)
        valid.append(support&(mass>=.8))
        mr=root/'source_model_warps'/f'cam{target:03d}'/f'src{cid:03d}'/'receipt.json';inputs[str(mr)]=sha(mr)
    yy,xx=np.mgrid[y0:y1,x0:x1];xy=np.stack([xx,yy],-1).astype(np.float32)
    center=np.asarray(rows[target]['transform_matrix'],np.float32)[:3,3]
    rays=unproject(np.ones(xx.shape,np.float32),rows[target],xy)-center
    rays/=np.maximum(np.linalg.norm(rays,axis=-1,keepdims=True),1e-8)
    offsets=[np.asarray(rows[c]['transform_matrix'],np.float32)[:3,3]-center for c in receipt['sources']]
    hashes={}
    for i,cid in enumerate(receipt['sources']):
        utilities=[];known=[]
        for j in range(len(receipt['sources'])):
            if i==j:continue
            support=valid[i]&valid[j]&(tangent_cosine(offsets[i],offsets[j],rays)<=-.25)
            u,k=restoration_utility(deltas[i],residuals[j],support);utilities.append(u);known.append(k)
        utility,count=witness_median(np.stack(utilities),np.stack(known))
        for key,value in [('utility',utility),('count',count.astype(np.uint8))]:
            p=out/f'src{cid:03d}_{key}.npy';np.save(p,value);hashes[p.name]=sha(p)
    write(out/'receipt.json',dict(target=target,parent=parent,sources=receipt['sources'],
        code_sha256=sha(Path(__file__)),inputs=inputs,hashes=hashes,
        band='G6-G16',witnesses_required=2,aggregation='median of all valid opposite witnesses',
        bracket_cosine_max=-.25,uses_target_rgb=False,witness_rgb_in_output=False,
        restoration_reference='zero added band versus witness photo minus frozen witness-view render'))
    log(root,'broad_witness_evidence_complete',camera=target,parent=parent);return out


def combine(root,parent,name,threshold):
    old=root/'candidates'/parent;out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    config={**read(old/'config.json'),'mode':'single_source_broad_witness','parent':parent,
            'broad_threshold':threshold,'broad_sigma':16.}
    assert config['sigma']==6. and config['filter_support']=='geometry'
    if (out/'config.json').exists():assert read(out/'config.json')==config
    else:write(out/'config.json',config)
    for target in [7,12,95,150]:
        if (out/f'cam{target:03d}_receipt.json').exists():continue
        receipt=read(old/f'cam{target:03d}_receipt.json');warp_root=Path(receipt['warp_root'])
        ed=evidence(root,target,parent);base=np.asarray(geometry(root,target)['rgb'])
        labels=np.load(old/f'cam{target:03d}_sources.npy',mmap_mode='r')
        deltas={};supports={};gates={};inputs={str(ed/'receipt.json'):sha(ed/'receipt.json')}
        for cid in receipt['sources']:
            if not np.any(labels==cid):continue
            d=warp_root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';r=read(d/'receipt.json')
            x0,y0,x1,y1=r['box'];sl=np.s_[y0:y1,x0:x1]
            a={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']}
            support=geometric_support(a,config)
            fine,broad,mass=split_bands(np.asarray(a['rgb']),support)
            fine_b,broad_b,_=split_bands(base[sl],support)
            gate=positive_gate(np.load(ed/f'src{cid:03d}_utility.npy'),np.load(ed/f'src{cid:03d}_count.npy'),support&(mass>=.8),threshold)
            deltas[cid]=(fine-fine_b)+gate[...,None]*(broad-broad_b)
            supports[cid]=support;gates[cid]=gate
        local,w=assemble(base[sl],deltas,supports,labels[sl],config)
        result=base.copy();result[sl]=local;weight=np.zeros(base.shape[:2],np.float32);weight[sl]=w
        assert np.isfinite(result).all() and np.array_equal(result[weight==0],base[weight==0])
        selected_gate=np.zeros(labels[sl].shape,np.float32)
        for cid,gate in gates.items():selected_gate[labels[sl]==cid]=gate[labels[sl]==cid]
        for suffix,value in [('',result),('_confidence',weight),('_broad_gate',selected_gate)]:np.save(out/f'cam{target:03d}{suffix}.npy',value)
        for suffix in ['sources.npy','sources.png','reasons.npy']:(out/f'cam{target:03d}_{suffix}').symlink_to(old/f'cam{target:03d}_{suffix}')
        image(out/f'cam{target:03d}.png',result);image(out/f'cam{target:03d}_transfer.png',np.repeat(weight[...,None],3,-1))
        for suffix in ['receipt.json','sources.npy']:
            p=old/f'cam{target:03d}_{suffix}';inputs[str(p)]=sha(p)
        for module in ['band_witness_luster_hybrid.py','supported_luster_hybrid.py']:
            p=Path(__file__).with_name(module);inputs[str(p)]=sha(p)
        write(out/f'cam{target:03d}_receipt.json',{**receipt,'config':config,'code_sha256':sha(Path(__file__)),
            'additional_input_hashes':inputs,'changed_fraction':float(np.any(result!=base,axis=-1).mean()),
            'broad_gate_sha256':sha(out/f'cam{target:03d}_broad_gate.npy')})
        log(root,'broad_witness_complete',camera=target,candidate=name)
