"""Reject contradictory middle-band structure while retaining fine single-photo detail."""
from pathlib import Path
import numpy as np
from luster_hybrid_common import geometry,image,log,read,sha,write
from supported_luster_hybrid import geometric_support
from band_witness_luster_hybrid import bands,assemble
from source_rgb_phase_quality_luster_hybrid import phase_cost


def structure_gate(photo_band, model_band, support, start):
    """Unknown or aligned structure stays; anti-correlated structure attenuates."""
    assert .5<=start<1
    cost=phase_cost(photo_band,model_band,support,sigma=4.)
    return np.where(support,np.clip((1-cost)/(1-start),0,1),0).astype(np.float32)


def combine(root,parent,name,start):
    old=root/'candidates'/parent;out=root/'candidates'/name;out.mkdir(parents=True,exist_ok=True)
    config={**read(old/'config.json'), 'mode':'single_source_structure_admission','parent':parent,'phase_admission_start':start}
    if (out/'config.json').exists():assert read(out/'config.json')==config
    else:write(out/'config.json',config)
    for target in [7,12,95,150]:
        if (out/f'cam{target:03d}_receipt.json').exists():continue
        receipt=read(old/f'cam{target:03d}_receipt.json');warp_root=Path(receipt['warp_root'])
        base=np.asarray(geometry(root,target)['rgb']);labels=np.load(old/f'cam{target:03d}_sources.npy',mmap_mode='r')
        deltas={};supports={};gates={};inputs={str(Path(__file__).with_name(m)):sha(Path(__file__).with_name(m)) for m in ['band_witness_luster_hybrid.py','source_rgb_phase_quality_luster_hybrid.py']}
        for cid in receipt['sources']:
            if not np.any(labels==cid):continue
            d=warp_root/'warps'/f'cam{target:03d}'/f'src{cid:03d}';r=read(d/'receipt.json');x0,y0,x1,y1=r['box'];sl=np.s_[y0:y1,x0:x1]
            a={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in r['arrays']};support=geometric_support(a,config)
            photo=bands(np.asarray(a['rgb']),support);baseline=bands(base[sl],support)
            gs=[support.astype(np.float32),structure_gate(photo[1],baseline[1],support,start)]
            inputs[str(d/'receipt.json')]=sha(d/'receipt.json')
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
        log(root,'structure_admission_complete',camera=target,candidate=name)
