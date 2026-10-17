"""Geometry-only coarse coverage of opposite-side TRAIN witnesses.

No RGB, evaluation mask or GT is read. This is a preflight, not proof of
source-to-witness visibility: that still requires a native reverse warp.
"""
import argparse
from pathlib import Path
import numpy as np
from luster_hybrid_common import read, write, sha, native_rows, geometry, unproject, assert_source
from transfer_luster_hybrid import source_score
from render_projection_luster_hybrid import project_block
from supported_luster_hybrid import geometric_support
from witness_benefit_luster_hybrid import tangent_cosine


def probe(root, target, step=24):
    rows=native_rows()
    config=read(root/'candidates/single_photometric_depth25/config.json')
    parent=root/'candidates/single_photometric_depth25_seam'
    receipt=parent/f'cam{target:03d}_receipt.json'
    sources=read(receipt)['sources']
    g=geometry(root,target)
    labels=np.load(parent/f'cam{target:03d}_sources.npy',mmap_mode='r')
    arrays={};inputs={str(receipt):sha(receipt)}
    for source in sources:
        assert_source(rows,target,source)
        d=root/'variants/cubic75/warps'/f'cam{target:03d}'/f'src{source:03d}'
        r=read(d/'receipt.json')
        if arrays:assert r['box']==box
        box=r['box']
        arrays[source]={k:np.asarray(np.load(d/f'{k}.npy',mmap_mode='r')[::step,::step])
                        for k in r['arrays'] if k!='rgb'}
        inputs[str(d/'receipt.json')]=sha(d/'receipt.json')
    x0,y0,x1,y1=box
    yy,xx=np.mgrid[y0:y1:step,x0:x1:step]
    xy=np.stack([xx,yy],-1).astype(np.float32)
    world=unproject(np.asarray(g['depth_z_q50'][yy,xx]),rows[target],xy)
    center=np.asarray(rows[target]['transform_matrix'],np.float32)[:3,3]
    rays=world-center
    rays/=np.maximum(np.linalg.norm(rays,axis=-1,keepdims=True),1e-8)
    offsets={c:np.asarray(rows[c]['transform_matrix'],np.float32)[:3,3]-center for c in sources}
    strict={c:geometric_support(arrays[c],config) for c in sources}
    wide={c:geometric_support(arrays[c],{**config,'max_score':2.}) for c in sources}
    witnesses=[c for c in sorted(rows) if c not in [7,12,95,150,target]
               and rows[c]['train'] and (root/'geometry'/f'cam{c:03d}'/'receipt.json').exists()]
    witness_scores={}
    for witness in witnesses:
        assert_source(rows,target,witness)
        wg=geometry(root,witness)
        # Zero-valued broadcast array: only geometry is sampled by project_block.
        wg={**wg,'rgb':np.broadcast_to(np.zeros(3,np.float32),(*wg['alpha'].shape,3))}
        block=project_block(rows[target],rows[witness],wg,xy,
            np.asarray(g['depth_z_q50'][yy,xx]),np.asarray(g['depth_z_q10'][yy,xx]),
            np.asarray(g['depth_z_q90'][yy,xx]),np.asarray(g['alpha'][yy,xx]))
        wide[witness]=geometric_support(block,{**config,'max_score':2.})
        witness_scores[witness]=block['score']
        offsets[witness]=np.asarray(rows[witness]['transform_matrix'],np.float32)[:3,3]-center
        path=root/'geometry'/f'cam{witness:03d}'/'receipt.json'
        inputs[str(path)]=sha(path)
    selected=labels[yy,xx]
    result={}
    for source in sources:
        eligible=strict[source]
        votes=[];supports={}
        for witness in witnesses:
            if witness==source:continue
            assert_source(rows,source,witness)
            pair_score,_=source_score(world,rows[source],rows[witness])
            opposite=tangent_cosine(offsets[source],offsets[witness],rays)<=-.25
            support=eligible&wide[witness]&opposite&(pair_score<=2.)
            supports[witness]=support
            weight=np.exp(-np.minimum(pair_score,100)-np.minimum(witness_scores[witness],100))
            votes.append((float(np.sum(np.where(support,weight,0))),witness))
        chosen=[c for v,c in sorted(votes,key=lambda x:(-x[0],x[1]))[:4] if v>0]
        count=sum((supports[c].astype(np.uint8) for c in chosen),start=np.zeros(eligible.shape,np.uint8))
        chosen_pixels=(selected==source)
        result[str(source)]=dict(witnesses=chosen,geometric_votes={str(c):v for v,c in votes},
            eligible_points=int(eligible.sum()),selected_points=int(chosen_pixels.sum()),
            two_witness_eligible_fraction=float(np.sum((count>=2)&eligible)/max(1,eligible.sum())),
            two_witness_selected_fraction=float(np.sum((count>=2)&chosen_pixels)/max(1,chosen_pixels.sum())))
    return dict(target=target,box=box,step=step,sources=result,inputs=inputs,
        code_sha256=sha(Path(__file__)),reads_rgb=False,uses_masks=False,considered_witnesses=witnesses,
        limitation='Coarse target-plane visibility only; source-to-native-witness visibility remains unverified.',
        bracket_cosine_max=-.25,confidence_max_score=2.,output_photo_max_score=config['max_score'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    results={str(c):probe(a.root,c) for c in [7,12,95,150]}
    write(a.root/'preflight/bracket_witness_all_geometry_coverage.json',results)
    for c,r in results.items():
        print(c,[(s,v['witnesses'],v['selected_points'],round(v['two_witness_selected_fraction'],3))
                 for s,v in r['sources'].items() if v['selected_points']>10],flush=True)
