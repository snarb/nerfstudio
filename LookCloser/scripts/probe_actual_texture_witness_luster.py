"""TRAIN-only confidence pilot for the actual source-to-target texture warp.

Unlike direct source-to-witness utility, first construct a single-source detail
candidate at the target's native q50 coordinates, then reproject that candidate
to a TRAIN witness. Only scalar confidence returns to the target. No target GT,
evaluation ROI or mask is read, and no candidate output is written.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from cubic75_luster_hybrid import sample
from luster_hybrid_common import (assert_source, geometry, log, native_rows,
    project, read, sha, source_path, unproject, write)
from risk_single_luster_hybrid import masked_low
from supported_luster_hybrid import geometric_support
from train_perceptual_utility_luster_hybrid import load_metric, metric_state_sha, perceptual_utility
from witness_native_restoration_luster_hybrid import sample_supported_scalar, median_cost


def run(root, output, target, sources, device):
    output.mkdir(parents=True, exist_ok=True)
    metric, metric_receipt = load_metric(output, device)
    rows = native_rows()
    config = read(root/'candidates/single_photometric_depth25/config.json')
    g = geometry(root, target)
    summary = []
    for source in sources:
        assert_source(rows, target, source)
        out = output/f'cam{target:03d}'/f'src{source:03d}'
        out.mkdir(parents=True, exist_ok=True)
        assert not (out/'receipt.json').exists(), 'Use a new diagnostic output'
        d = root/'variants/cubic75/warps'/f'cam{target:03d}'/f'src{source:03d}'
        tr = read(d/'receipt.json'); box = tr['box']; x0,y0,x1,y1 = box
        sl = np.s_[y0:y1,x0:x1]
        a = {k:np.load(d/f'{k}.npy',mmap_mode='r') for k in tr['arrays']}
        support_t = geometric_support(a,config)
        base = np.asarray(g['rgb'][sl])
        low_source,_ = masked_low(a['rgb'],support_t,config['sigma'])
        low_base,_ = masked_low(base,support_t,config['sigma'])
        candidate = np.clip(base+(a['rgb']-low_source)-(base-low_base),0,1)
        crop_geometry = {k:g[k][sl] for k in ['depth_z_q10','depth_z_q50','depth_z_q90']}
        # Cubic sampler requires every nonzero tap to be geometrically eligible
        # for the target's chosen source, not merely valid target geometry.
        crop_geometry['alpha'] = support_t.astype(np.float32)
        old_path = root/'witness_native_restoration'/f'cam{source:03d}'/'receipt.json'
        witnesses = read(old_path)['witnesses']
        inputs = {str(p):sha(p) for p in [d/'receipt.json', old_path, metric_receipt,
            root/'geometry'/f'cam{target:03d}'/'receipt.json']}
        utilities=[]; validity=[]; witness_rows=[]
        for witness in witnesses:
            assert_source(rows,target,witness); assert_source(rows,witness,source)
            wd = root/'variants/cubic75/warps'/f'cam{witness:03d}'/f'src{source:03d}'
            wr = read(wd/'receipt.json'); wbox=wr['box']; wx0,wy0,wx1,wy1=wbox
            wa={k:np.load(wd/f'{k}.npy',mmap_mode='r') for k in wr['arrays']}
            variant=root/'variants/target_witness_confidence60/frozen_render_projections'
            forward=variant/f'cam{witness:03d}'/f'render{target:03d}'
            reverse=variant/f'cam{target:03d}'/f'render{witness:03d}'
            assert read(forward/'receipt.json')['box']==wbox
            assert read(reverse/'receipt.json')['box']==box
            expected=np.load(forward/'rgb.npy',mmap_mode='r')
            supported=np.array(np.load(forward/'support.npy',mmap_mode='r'))
            supported &= geometric_support(wa,config)
            baseline_w=np.empty_like(expected); candidate_w=np.empty_like(expected)
            wg=geometry(root,witness)
            maximum_baseline_difference=0.
            for y in range(wy0,wy1,128):
                stop=min(y+128,wy1); yy,xx=np.mgrid[y:stop,wx0:wx1]
                xy=np.stack([xx,yy],-1).astype(np.float32)
                uv,z=project(unproject(np.asarray(wg['depth_z_q50'][y:stop,wx0:wx1]),rows[witness],xy),rows[target])
                uv=uv-np.array([x0,y0],np.float32)
                b,good,alpha,_=sample(base,crop_geometry,uv,z)
                c,cgood,calpha,_=sample(candidate,crop_geometry,uv,z)
                dest=np.s_[y-wy0:stop-wy0]
                supported[dest] &= good & cgood & (alpha>=1) & (calpha>=1)
                if supported[dest].any():
                    delta=np.abs(b-expected[dest])[supported[dest]]
                    maximum_baseline_difference=max(maximum_baseline_difference,float(delta.max()))
                baseline_w[dest]=b; candidate_w[dest]=c
            assert maximum_baseline_difference<2e-6, maximum_baseline_difference
            photo_path=source_path(witness)
            assert sha(photo_path)==read(root/'preflight/manifest.json')['source_hashes'][str(photo_path)]
            photo=np.asarray(Image.open(photo_path).convert('RGB'),np.float32)[wy0:wy1,wx0:wx1]/255
            utility,known=perceptual_utility(candidate_w,photo,baseline_w,supported,metric)
            reverse_support=np.load(reverse/'support.npy',mmap_mode='r')
            result=np.zeros(reverse_support.shape,np.float32); valid=np.zeros(reverse_support.shape,bool)
            for y in range(y0,y1,128):
                stop=min(y+128,y1); yy,xx=np.mgrid[y:stop,x0:x1]
                xy=np.stack([xx,yy],-1).astype(np.float32)
                uv,_=project(unproject(np.asarray(g['depth_z_q50'][y:stop,x0:x1]),rows[target],xy),rows[witness])
                value,good=sample_supported_scalar(utility,known,uv,wbox)
                dest=np.s_[y-y0:stop-y0]; valid[dest]=good&reverse_support[dest]
                result[dest]=np.where(valid[dest],value,0)
            utilities.append(result);validity.append(valid)
            witness_rows.append(dict(witness=witness,known_pixels=int(valid.sum()),
                baseline_equivalence_max_abs=maximum_baseline_difference))
            for p in [photo_path,wd/'receipt.json',forward/'receipt.json',reverse/'receipt.json']:
                inputs[str(p)]=sha(p)
        cost,count=median_cost(np.stack(utilities),np.stack(validity))
        for key,value in [('cost',cost),('count',count)]:np.save(out/f'{key}.npy',value)
        for module in ['cubic75_luster_hybrid.py','luster_hybrid_common.py','risk_single_luster_hybrid.py',
                       'supported_luster_hybrid.py','train_perceptual_utility_luster_hybrid.py',
                       'witness_native_restoration_luster_hybrid.py']:
            p=Path(__file__).with_name(module);inputs[str(p)]=sha(p)
        receipt=dict(target=target,source=source,box=box,config=config,witnesses=witness_rows,
            inputs=inputs,code_sha256=sha(Path(__file__)),
            hashes={k:sha(out/k) for k in ['cost.npy','count.npy']},
            known_fraction=float(np.mean(count>=2)),geometry='frozen camera-z q50',
            uses_target_gt=False,uses_masks=False,lookup_shift=False,
            note='Actual source-to-target detail candidate reprojected to TRAIN witness; confidence only.')
        write(out/'receipt.json',receipt);summary.append(receipt)
        log(root,'actual_texture_witness_pilot_complete',target=target,source=source,
            known_fraction=receipt['known_fraction'])
    assert metric_state_sha(metric)==read(metric_receipt)['metric_state_sha256']
    write(output/'summary.json',dict(rows=summary,metric_unchanged=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--target',type=int,default=7)
    p.add_argument('--sources',type=int,nargs='+',default=[8,6]);p.add_argument('--device',default='cpu')
    a=p.parse_args();run(a.root,a.output,a.target,a.sources,a.device)
