"""Screened gradient transfer within fixed single-photo patches.

Solve (L + anchor I) delta = L (selected_photo - LookCloser), where L has
edges only between pixels of the same selected photograph. No cross-source
RGB edge or target GT enters the solve. Retain the parent's boundary fade.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import LinearOperator,cg
from luster_hybrid_common import geometry,image,read,write,sha,log


def solve_delta(base,texture,labels,anchor,tolerance=1e-5):
    assert anchor>0
    valid=labels>=0;horizontal=valid[:,:-1]&valid[:,1:]&(labels[:,:-1]==labels[:,1:]);vertical=valid[:-1]&valid[1:]&(labels[:-1]==labels[1:])
    shape=labels.shape;degree=np.zeros(shape,np.float32)
    degree[:,:-1]+=horizontal;degree[:,1:]+=horizontal;degree[:-1]+=vertical;degree[1:]+=vertical
    def laplace(x):
        y=degree*x
        y[:,:-1]-=horizontal*x[:,1:];y[:,1:]-=horizontal*x[:,:-1]
        y[:-1]-=vertical*x[1:];y[1:]-=vertical*x[:-1]
        return y
    def apply(x):
        x=x.reshape(shape);return (laplace(x)+anchor*x).reshape(-1)
    op=LinearOperator((labels.size,labels.size),matvec=apply,dtype=np.float32)
    inv=(degree+anchor).reshape(-1)
    preconditioner=LinearOperator(op.shape,matvec=lambda x:x/inv,dtype=np.float32)
    delta=np.zeros_like(base);reports=[]
    for channel in range(3):
        rhs=laplace(np.where(valid,texture[...,channel]-base[...,channel],0.)).reshape(-1)
        steps=[0]
        def count(_):steps[0]+=1
        x,info=cg(op,rhs,rtol=tolerance,atol=1e-8,maxiter=400,M=preconditioner,callback=count)
        relative=float(np.linalg.norm(apply(x)-rhs)/max(np.linalg.norm(rhs),1e-12))
        assert info==0 and relative<=tolerance*5, (info,relative)
        delta[...,channel]=x.reshape(shape);reports.append(dict(iterations=steps[0],relative_residual=relative))
    assert np.all(delta[~valid]==0)
    return delta,reports


def combine(root,parent,name,anchor):
    old=root/'candidates'/parent;out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    config={**read(old/'config.json'),'mode':'single_source_screened_poisson','parent':parent,'poisson_anchor':anchor,'poisson_rtol':1e-5}
    if (out/'config.json').exists():assert read(out/'config.json')==config
    else:write(out/'config.json',config)
    for target in [7,12,95,150]:
        if (out/f'cam{target:03d}_receipt.json').exists():continue
        receipt=read(old/f'cam{target:03d}_receipt.json');warp_root=Path(receipt.get('warp_root',root));base=np.asarray(geometry(root,target)['rgb']);labels=np.load(old/f'cam{target:03d}_sources.npy',mmap_mode='r');weight=np.load(old/f'cam{target:03d}_confidence.npy',mmap_mode='r')
        first=warp_root/'warps'/f'cam{target:03d}'/f"src{receipt['sources'][0]:03d}";x0,y0,x1,y1=read(first/'receipt.json')['box'];sl=np.s_[y0:y1,x0:x1];lab=labels[sl];texture=base[sl].copy()
        for cid in receipt['sources']:
            photo=np.load(warp_root/'warps'/f'cam{target:03d}'/f'src{cid:03d}'/'rgb.npy',mmap_mode='r');chosen=lab==cid;texture[chosen]=photo[chosen]
        delta,solver=solve_delta(base[sl],texture,lab,anchor,config['poisson_rtol']);result=base.copy();result[sl]=np.clip(base[sl]+weight[sl][...,None]*delta,0,1)
        assert np.isfinite(result).all() and np.array_equal(result[weight==0],base[weight==0])
        np.save(out/f'cam{target:03d}.npy',result);image(out/f'cam{target:03d}.png',result)
        for suffix in ['sources.npy','confidence.npy','reasons.npy','sources.png','transfer.png']:(out/f'cam{target:03d}_{suffix}').symlink_to(old/f'cam{target:03d}_{suffix}')
        inputs={str(old/f'cam{target:03d}_{suffix}'):sha(old/f'cam{target:03d}_{suffix}') for suffix in ['receipt.json','sources.npy','confidence.npy']}
        write(out/f'cam{target:03d}_receipt.json',{**receipt,'config':config,'code_sha256':sha(Path(__file__)),'additional_input_hashes':inputs,'solver':solver,'changed_fraction':float(np.any(result!=base,axis=-1).mean())})
        log(root,'single_poisson_complete',camera=target,candidate=name,solver=solver)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',default='single_cubic_fixed');p.add_argument('--name',required=True);p.add_argument('--anchor',type=float,required=True);a=p.parse_args();combine(a.root,a.parent,a.name,a.anchor)
