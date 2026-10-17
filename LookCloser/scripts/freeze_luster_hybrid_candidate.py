"""Audit a complete candidate and bind it to its actual warp variant and outputs."""
import argparse
from pathlib import Path
import time
import numpy as np
from luster_hybrid_common import (DIGEST,TARGETS,allowed_sources,geometry,native_rows,read,sha,source_path,write)


def freeze(root,name,warp_root,deep=False,require_single_source=False):
    rows=native_rows();candidate=root/'candidates'/name
    manifest=read(root/'preflight/manifest.json');config=read(candidate/'config.json')
    inputs={};checks=[];seen_geometry=set()
    state_hash=read(root/'geometry/cam007/receipt.json')['state_hash']
    inventory={p.name:sha(p) for p in sorted((root/'code/LookCloser/scripts').glob('*luster_hybrid*.py'))}
    def bind(path):
        path=Path(path).resolve()
        if str(path) not in inputs:inputs[str(path)]=sha(path)
    for cid in TARGETS:
        receipt=read(candidate/f'cam{cid:03d}_receipt.json')
        assert receipt['config']==config
        assert receipt['code_sha256'] in inventory.values(), 'Candidate generator changed since creation'
        for path,digest in receipt.get('additional_input_hashes',{}).items():
            assert sha(path)==digest;bind(path)
            if 'source_model_warps' in path:
                mr=read(path);assert mr['code_sha256'] in inventory.values()
                cached=Path(path).parent/'rgb.npy'
                if deep:assert sha(cached)==mr['rgb_sha256'];bind(cached)
        allowed=set(allowed_sources(rows,cid));assert set(receipt['sources'])<=allowed
        baseline=geometry(root,cid)['rgb'];pred=np.load(candidate/f'cam{cid:03d}.npy',mmap_mode='r')
        weight=np.load(candidate/f'cam{cid:03d}_confidence.npy',mmap_mode='r')
        labels=np.load(candidate/f'cam{cid:03d}_sources.npy',mmap_mode='r')
        assert pred.shape==baseline.shape and np.isfinite(pred).all() and pred.min()>=0 and pred.max()<=1
        assert set(map(int,np.unique(labels[labels>=0])))<=allowed
        assert np.array_equal(pred[weight==0],baseline[weight==0])
        for source in receipt['sources']:
            d=warp_root/'warps'/f'cam{cid:03d}'/f'src{source:03d}'
            r=read(d/'receipt.json');assert r['source']==source and r['target']==cid
            bind(d/'receipt.json')
            assert r['target_geometry_sha256']==sha(root/'geometry'/f'cam{cid:03d}'/'receipt.json')
            assert r['source_geometry_sha256']==sha(root/'geometry'/f'cam{source:03d}'/'receipt.json')
            assert r['source_sha256']==manifest['source_hashes'][str(source_path(source))]
            if deep:assert sha(source_path(source))==r['source_sha256']
            if 'parent_warp_receipt_sha256' in r:
                parent=root/'warps'/f'cam{cid:03d}'/f'src{source:03d}'/'receipt.json'
                assert sha(parent)==r['parent_warp_receipt_sha256'];bind(parent)
            if deep:
                for field in r['arrays']:bind(d/f'{field}.npy')
            if 'source_foreground_receipt_sha256' in r:
                family='source_foreground_noise' if 'foreground_noise' in str(warp_root) else 'source_foreground'
                fg=root/family/f'cam{source:03d}'/'receipt.json'
                assert sha(fg)==r['source_foreground_receipt_sha256'];bind(fg)
                fr=read(fg);assert sha(fr['background']['path'])==fr['background']['sha256']
            for c in [cid,source]:
                if c in seen_geometry:continue
                seen_geometry.add(c);g=root/'geometry'/f'cam{c:03d}';gr=read(g/'receipt.json')
                assert gr['state']=='complete' and gr['checkpoint_sha256']==DIGEST
                assert gr['state_hash']==state_hash, 'Frozen model/grids changed'
                bind(g/'receipt.json')
                if deep:
                    for field,value in gr['hashes'].items():assert sha(g/f'{field}.npy')==value
        if require_single_source:
            assert receipt.get('texture_source_count_per_pixel')==1
            assert receipt.get('lookup_shift') is False
        single_verified=False
        if receipt.get('texture_source_count_per_pixel')==1:
            from transfer_luster_hybrid import merge_details
            assert warp_root.resolve()==root.resolve(), 'Single-source audit requires exact original projections'
            assert receipt.get('lookup_shift') is False
            first=warp_root/'warps'/f'cam{cid:03d}'/f"src{receipt['sources'][0]:03d}"
            x0,y0,x1,y1=read(first/'receipt.json')['box'];sl=np.s_[y0:y1,x0:x1]
            local_labels=np.asarray(labels[sl]);texture=np.asarray(baseline[sl]).copy()
            for source in receipt['sources']:
                d=warp_root/'warps'/f'cam{cid:03d}'/f'src{source:03d}'
                selected=local_labels==source
                if not selected.any():continue
                photo=np.load(d/'rgb.npy',mmap_mode='r');texture[selected]=photo[selected]
                for field,limit,lower in [('alpha',config['alpha'],True),('resolution_ratio',config['min_resolution'],True),('depth_error',config['depth_error'],False),('roundtrip',config['roundtrip'],False),('uncertainty',config['uncertainty'],False)]:
                    values=np.load(d/f'{field}.npy',mmap_mode='r')[selected]
                    assert np.all(values>=limit) if lower else np.all(values<=limit)
            rebuilt,rebuilt_weight=merge_details(np.asarray(baseline[sl]),texture,local_labels,(local_labels>=0).astype(np.float32),config['mode'],config['sigma'],config['gain'])
            np.testing.assert_array_equal(rebuilt,pred[sl])
            np.testing.assert_array_equal(rebuilt_weight,weight[sl])
            single_verified=True
        pairs=candidate/f'cam{cid:03d}_blend_provenance.npz'
        if require_single_source:assert not pairs.exists()
        if pairs.exists():
            with np.load(pairs) as blend:
                for field in ['first','second']:
                    values=blend[field];assert set(map(int,np.unique(values[values>=0])))<=allowed
                assert np.isfinite(blend['first_weight']).all()
                assert blend['first_weight'].min()>=0 and blend['first_weight'].max()<=1
        checks.append(dict(camera=cid,exact_fallback=True,source_exclusion=True,single_source_reconstruction_verified=single_verified,
                           changed_fraction=float(np.any(pred!=baseline,axis=-1).mean())))
    if deep:
        for path,digest in manifest['inputs'].items():assert sha(path)==digest
        for rel,digest in manifest['runtime_hashes'].items():assert sha(root/'runtime'/rel)==digest
    outputs={str(p.relative_to(candidate)):sha(p) for p in sorted(candidate.iterdir()) if p.is_file() and p.name!='provenance.json'}
    report=dict(candidate=name,created=time.time(),warp_root=str(warp_root.resolve()),
         checkpoint_sha256=DIGEST,checks=checks,deep=deep,inputs=inputs,outputs=outputs,
         uses_target_ground_truth_for_prediction=False,uses_target_evaluation_masks_for_prediction=False,
         uses_derived_source_foreground_masks='foreground' in str(warp_root),
         uses_classical_lookup_alignment='align' in str(warp_root) or bool(receipt.get('registration')),
         uses_source_model_residual=config.get('mode')=='source_model_residual',
         code=inventory)
    write(candidate/'provenance.json',report)
    print(name,': source exclusion, finite outputs and exact fallback verified; deep=',deep,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--warp-root',type=Path)
    p.add_argument('--deep',action='store_true');p.add_argument('--require-single-source',action='store_true');a=p.parse_args()
    freeze(a.root,a.name,a.warp_root or a.root,a.deep,a.require_single_source)
