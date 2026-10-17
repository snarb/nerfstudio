"""Admit two detail bands using TRAIN evidence, with one photo supplying RGB."""
from pathlib import Path
import cv2
import numpy as np
from luster_hybrid_common import assert_source, geometry, image, log, native_rows, read, sha, unproject, write
from supported_luster_hybrid import geometric_support
from transfer_luster_hybrid import gaussian
from witness_benefit_luster_hybrid import pair_benefit, tangent_cosine, update_best
cv2.setNumThreads(4)


def bands(rgb, support):
    sf=support.astype(np.float32)
    def low(s):
        mass=gaussian(sf,s)
        return gaussian(rgb*sf[...,None],s)/np.maximum(mass[...,None],1e-8)
    two=low(2.);six=low(6.)
    return rgb-two,two-six


def admission(utility, support, margin):
    """No demonstrated harm keeps full detail; two adverse witnesses attenuate."""
    assert margin>0
    gate=np.clip(1+utility/margin,0,1)
    mass=gaussian(support.astype(np.float32),2.)
    return np.where(support,np.clip(gaussian(gate*support,2.)/np.maximum(mass,1e-8),0,1),0).astype(np.float32)


def evidence(root,target,parent):
    old=root/'candidates'/parent;receipt=read(old/f'cam{target:03d}_receipt.json');config=receipt['config']
    out=root/'band_witness'/parent/f'cam{target:03d}';out.mkdir(parents=True,exist_ok=True)
    if (out/'receipt.json').exists():
        r=read(out/'receipt.json');assert r['code_sha256']==sha(Path(__file__))
        assert all(sha(Path(k))==v for k,v in r['inputs'].items())
        assert all(sha(out/k)==v for k,v in r['hashes'].items());return out
    rows=native_rows();warp_root=Path(receipt['warp_root']);base=geometry(root,target)['rgb']
    photo=[];baseline=[];valid=[];inputs={str(old/f'cam{target:03d}_receipt.json'):sha(old/f'cam{target:03d}_receipt.json')}
    luminance=np.array([.2126,.7152,.0722],np.float32)
    for cid in receipt['sources']:
        assert_source(rows,target,cid);d=warp_root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';r=read(d/'receipt.json')
        inputs[str(d/'receipt.json')]=sha(d/'receipt.json')
        a={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']};support=geometric_support(a,config)
        x0,y0,x1,y1=r['box'];sl=np.s_[y0:y1,x0:x1]
        photo.append([np.einsum('...c,c->...',b,luminance) for b in bands(np.asarray(a['rgb']),support)])
        baseline.append([np.einsum('...c,c->...',b,luminance) for b in bands(np.asarray(base[sl]),support)])
        valid.append(support)
    yy,xx=np.mgrid[y0:y1,x0:x1];xy=np.stack([xx,yy],-1).astype(np.float32)
    center=np.asarray(rows[target]['transform_matrix'],np.float32)[:3,3]
    rays=unproject(np.ones(xx.shape,np.float32),rows[target],xy)-center
    rays/=np.maximum(np.linalg.norm(rays,axis=-1,keepdims=True),1e-8)
    offsets=[np.asarray(rows[c]['transform_matrix'],np.float32)[:3,3]-center for c in receipt['sources']]
    best=[[np.full(xx.shape,-2.,np.float32) for _ in range(2)] for _ in photo]
    second=[[x.copy() for x in item] for item in best]
    for i in range(len(photo)):
        for j in range(i+1,len(photo)):
            support=valid[i]&valid[j]&(tangent_cosine(offsets[i],offsets[j],rays)<=-.25)
            for band in range(2):
                # Compare against the TARGET baseline, normalized over each
                # candidate's unchanged geometric support; witnesses supply no RGB.
                utility,known=pair_benefit(photo[i][band],photo[j][band],baseline[i][band],support)
                update_best(best[i][band],second[i][band],utility,known)
                utility,known=pair_benefit(photo[j][band],photo[i][band],baseline[j][band],support)
                update_best(best[j][band],second[j][band],utility,known)
    hashes={}
    for i,cid in enumerate(receipt['sources']):
        for b in range(2):
            known=second[i][b]>=-1.
            utility=np.where(known,(best[i][b]+second[i][b])*.5,0).astype(np.float32)
            p=out/f'src{cid:03d}_band{b}_utility.npy';np.save(p,utility);hashes[p.name]=sha(p)
    write(out/'receipt.json',dict(target=target,parent=parent,sources=receipt['sources'],code_sha256=sha(Path(__file__)),inputs=inputs,hashes=hashes,
          bands=['I-G2','G2-G6'],witnesses_required=2,bracket_cosine_max=-.25,uses_target_rgb=False,witness_rgb_in_output=False,missing_evidence_utility=0.))
    log(root,'band_witness_evidence_complete',camera=target,parent=parent);return out


def assemble(base,deltas,supports,labels,config):
    """Single-source correction with seam checks on the actual admitted bands."""
    selected=np.zeros_like(base)
    for cid,delta in deltas.items():selected[labels==cid]=delta[labels==cid]
    kernel=np.ones((7,7),np.uint8)
    full=(labels>=0)&(cv2.erode((labels>=0).astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0)
    for cid,delta in deltas.items():
        near=(cv2.dilate((labels==cid).astype(np.uint8),kernel)>0)&(labels!=cid)&(labels>=0)
        difference=cv2.dilate(np.max(np.abs(delta-selected),axis=-1),np.ones((3,3),np.uint8))
        full[near&~(supports[cid]&(difference<=config['seam_tolerance']))]=False
    weight=np.zeros(labels.shape,np.float32)
    for cid in deltas:
        chosen=labels==cid;distance=cv2.distanceTransform(chosen.astype(np.uint8),cv2.DIST_L2,5)
        weight[chosen]=np.minimum(distance[chosen]/config['seam_fade'],1)*config['gain']
    weight[full]=config['gain'];result=base.copy();chosen=weight>0
    result[chosen]=np.clip(base[chosen]+weight[chosen,None]*selected[chosen],0,1)
    return result,weight


def combine(root,parent,name,margin,mid_only=False):
    old=root/'candidates'/parent;out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    config={**read(old/'config.json'), 'mode':'single_source_band_witness','parent':parent,'band_margin':margin,'mid_only':mid_only}
    if (out/'config.json').exists():assert read(out/'config.json')==config
    else:write(out/'config.json',config)
    for target in [7,12,95,150]:
        if (out/f'cam{target:03d}_receipt.json').exists():continue
        receipt=read(old/f'cam{target:03d}_receipt.json');warp_root=Path(receipt['warp_root']);ed=evidence(root,target,parent)
        base=np.asarray(geometry(root,target)['rgb']);labels=np.load(old/f'cam{target:03d}_sources.npy',mmap_mode='r')
        deltas={};supports={};gates={};inputs={str(ed/'receipt.json'):sha(ed/'receipt.json')}
        for cid in receipt['sources']:
            if not np.any(labels==cid):continue
            d=warp_root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';r=read(d/'receipt.json');x0,y0,x1,y1=r['box'];sl=np.s_[y0:y1,x0:x1]
            a={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']};support=geometric_support(a,config)
            photo=bands(np.asarray(a['rgb']),support);baseline=bands(base[sl],support)
            gs=[admission(np.load(ed/f'src{cid:03d}_band{b}_utility.npy'),support,margin) for b in range(2)]
            if mid_only:gs[0]=support.astype(np.float32)
            deltas[cid]=sum(g[...,None]*(p-b) for g,p,b in zip(gs,photo,baseline));supports[cid]=support;gates[cid]=gs
        local,local_weight=assemble(base[sl],deltas,supports,labels[sl],config)
        result=base.copy();result[sl]=local;weight=np.zeros(base.shape[:2],np.float32);weight[sl]=local_weight
        assert np.isfinite(result).all() and np.array_equal(result[weight==0],base[weight==0])
        np.save(out/f'cam{target:03d}.npy',result);np.save(out/f'cam{target:03d}_confidence.npy',weight)
        selected_gates=np.zeros((*labels[sl].shape,2),np.float32)
        for cid,gs in gates.items():
            choose=labels[sl]==cid
            for b in range(2):selected_gates[...,b][choose]=gs[b][choose]
        np.save(out/f'cam{target:03d}_band_gates.npy',selected_gates)
        for suffix in ['sources.npy','sources.png','reasons.npy']:(out/f'cam{target:03d}_{suffix}').symlink_to(old/f'cam{target:03d}_{suffix}')
        image(out/f'cam{target:03d}.png',result);image(out/f'cam{target:03d}_transfer.png',np.repeat(weight[...,None],3,-1))
        for suffix in ['receipt.json','sources.npy']:inputs[str(old/f'cam{target:03d}_{suffix}')]=sha(old/f'cam{target:03d}_{suffix}')
        write(out/f'cam{target:03d}_receipt.json',{**receipt,'config':config,'code_sha256':sha(Path(__file__)),'additional_input_hashes':inputs,
              'changed_fraction':float(np.any(result!=base,axis=-1).mean()),'band_gates_sha256':sha(out/f'cam{target:03d}_band_gates.npy')})
        log(root,'band_witness_complete',camera=target,candidate=name)
