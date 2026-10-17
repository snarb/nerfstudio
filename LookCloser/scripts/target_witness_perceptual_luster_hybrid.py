"""Native TRAIN LPIPS utility relative to the actual target frozen render.

The target render is reprojected only to evaluate scalar confidence. Its GT is
never read. Final photo sampling stays at the target's original frozen q50.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from luster_hybrid_common import assert_source, geometry, log, native_rows, project, read, sha, source_path, unproject, write
from source_surface_confidence_luster_hybrid import EXCLUDED
from train_perceptual_utility_luster_hybrid import perceptual_utility
from witness_native_restoration_luster_hybrid import sample_supported_scalar, median_cost
from render_projection_luster_hybrid import render_projection as project_render
from risk_single_luster_hybrid import masked_low
from supported_luster_hybrid import geometric_support


def render_projection(root, observer, render_view, box, config):
    """Wider angular preference for scalar render comparison, same visibility."""
    variant=root/'variants/target_witness_confidence60'
    variant.mkdir(parents=True,exist_ok=True)
    for key in ['geometry','logs']:
        path=variant/key
        if not path.exists():path.symlink_to(root/key,target_is_directory=True)
        assert path.resolve()==(root/key).resolve()
    return project_render(variant,observer,render_view,box,{**config,'max_score':2.})


def cached_utility(root,target,source,config):
    """Read-only source ranker accessor; no metric/GPU work is launched here."""
    assert_source(native_rows(),target,source)
    out=root/'target_witness_perceptual_projected'/f'cam{target:03d}'/f'src{source:03d}'
    r=read(out/'receipt.json')
    assert r['code_sha256']==sha(Path(__file__)) and r['target']==target and r['source']==source
    assert r['gates']=={k:config[k] for k in ['alpha','depth_error','roundtrip','uncertainty','max_score','min_resolution']}
    assert all(sha(out/k)==v for k,v in r['hashes'].items())
    return out


def projected_utility(root,target,source,config,metric,metric_receipt):
    rows=native_rows();assert_source(rows,target,source)
    dependency=root/'witness_native_restoration'/f'cam{source:03d}'/'receipt.json'
    dr=read(dependency)
    gates={k:config[k] for k in ['alpha','depth_error','roundtrip','uncertainty','max_score','min_resolution']}
    assert dr['source']==source and dr['gates']==gates
    witnesses=dr['witnesses']
    assert len(witnesses)>=2 and len(set(witnesses))==len(witnesses)
    assert not set(witnesses)&set(EXCLUDED) and source not in witnesses
    out=root/'target_witness_perceptual_projected'/f'cam{target:03d}'/f'src{source:03d}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'receipt.json').exists():
        r=read(out/'receipt.json')
        assert r['code_sha256']==sha(Path(__file__)) and r['gates']==gates
        assert r['inputs'][str(dependency)]==sha(dependency)
        assert r['inputs'][str(metric_receipt)]==sha(metric_receipt)
        assert all(sha(out/k)==v for k,v in r['hashes'].items());return out
    wr=root/'warps'/f'cam{target:03d}'/f'src{source:03d}'/'receipt.json'
    box=read(wr)['box'];x0,y0,x1,y1=box;g=geometry(root,target)
    inputs={str(p):sha(p) for p in [dependency,wr,metric_receipt,root/'geometry'/f'cam{target:03d}'/'receipt.json']}
    for module in ['luster_hybrid_common.py','source_surface_confidence_luster_hybrid.py',
                   'train_perceptual_utility_luster_hybrid.py','witness_native_restoration_luster_hybrid.py',
                   'render_projection_luster_hybrid.py',
                   'risk_single_luster_hybrid.py','supported_luster_hybrid.py']:
        p=Path(__file__).with_name(module);inputs[str(p)]=sha(p)
    utilities=[];validity=[]
    for witness in witnesses:
        assert_source(rows,target,witness);assert_source(rows,witness,source)
        d=root/'variants/cubic75/warps'/f'cam{witness:03d}'/f'src{source:03d}'
        sr=read(d/'receipt.json');assert sr['target']==witness and sr['source']==source
        wbox=sr['box'];wx0,wy0,wx1,wy1=wbox
        arrays={k:np.load(d/f'{k}.npy',mmap_mode='r') for k in sr['arrays']}
        forward=render_projection(root,witness,target,wbox,config)
        reverse=render_projection(root,target,witness,box,config)
        support=geometric_support(arrays,config)&np.load(forward/'support.npy',mmap_mode='r')
        native=source_path(witness)
        assert sha(native)==read(root/'preflight/manifest.json')['source_hashes'][str(native)]
        photo=np.asarray(Image.open(native).convert('RGB'),np.float32)[wy0:wy1,wx0:wx1]/255
        baseline=np.load(forward/'rgb.npy',mmap_mode='r')
        source_low,_=masked_low(arrays['rgb'],support,6.)
        baseline_low,_=masked_low(baseline,support,6.)
        candidate=np.clip(baseline+(arrays['rgb']-source_low)-(baseline-baseline_low),0,1)
        utility,known=perceptual_utility(candidate,photo,baseline,support,metric)
        reverse_support=np.load(reverse/'support.npy',mmap_mode='r')
        utility_out=np.zeros(reverse_support.shape,np.float32)
        valid=np.zeros(reverse_support.shape,bool)
        for y in range(y0,y1,128):
            stop=min(y+128,y1);yy,xx=np.mgrid[y:stop,x0:x1]
            xy=np.stack([xx,yy],-1).astype(np.float32)
            uv,_=project(unproject(np.asarray(g['depth_z_q50'][y:stop,x0:x1]),rows[target],xy),rows[witness])
            sampled_utility,good=sample_supported_scalar(utility,known,uv,wbox)
            sl=np.s_[y-y0:stop-y0];valid[sl]=good&reverse_support[sl]
            utility_out[sl]=np.where(valid[sl],sampled_utility,0)
        utilities.append(utility_out);validity.append(valid)
        for p in [native,d/'receipt.json',forward/'receipt.json',reverse/'receipt.json']:
            inputs[str(p)]=sha(p)
        for key in sr['arrays']:inputs[str(d/f'{key}.npy')]=sha(d/f'{key}.npy')
    valid=np.stack(validity)
    cost,count=median_cost(np.stack(utilities),valid)
    for key,value in [('cost',cost),('count',count)]:np.save(out/f'{key}.npy',value)
    write(out/'receipt.json',dict(target=target,source=source,witnesses=witnesses,box=box,gates=gates,
        inputs=inputs,code_sha256=sha(Path(__file__)),hashes={k:sha(out/k) for k in ['cost.npy','count.npy']},
        render_confidence_max_score=2.,photo_max_score=config['max_score'],confidence_frame='native TRAIN witness',baseline='actual target frozen render reprojected for confidence only',
        detail='I-G6 RGB, candidate recomposed on target baseline in native TRAIN witness frame',
        metric='frozen evaluation LPIPS AlexNet spatial',support_sigma=16.,support_mass=.8,utility_floor=.001,unknown_cost=.5,
        geometry='frozen camera-z q50',uses_target_gt=False,uses_masks=False,witness_rgb_in_output=False,
        target_render_warp_in_output=False,lookup_shift=False))
    log(root,'target_witness_perceptual_complete',target=target,source=source,known_fraction=float(np.mean(count>=2)))
    return out
