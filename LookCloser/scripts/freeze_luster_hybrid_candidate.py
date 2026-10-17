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
            if Path(path).name=='receipt.json':
                extra=read(path)
                if 'code_sha256' in extra:assert extra['code_sha256'] in inventory.values()
                for ip,ih in extra.get('inputs',{}).items():assert sha(ip)==ih;bind(ip)
                if deep:
                    for key,value in extra.get('hashes',{}).items():
                        fp=Path(path).parent/key
                        if not fp.exists():fp=fp.with_suffix('.npy')
                        assert sha(fp)==value;bind(fp)
            if 'source_model_warps' in path:
                mr=read(path);assert mr['code_sha256'] in inventory.values()
                cached=Path(path).parent/'rgb.npy'
                if deep:assert sha(cached)==mr['rgb_sha256'];bind(cached)
        allowed=set(allowed_sources(rows,cid));assert set(receipt['sources'])<=allowed
        baseline=geometry(root,cid)['rgb'];pred=np.load(candidate/f'cam{cid:03d}.npy',mmap_mode='r')
        weight=np.load(candidate/f'cam{cid:03d}_confidence.npy',mmap_mode='r')
        labels=np.load(candidate/f'cam{cid:03d}_sources.npy',mmap_mode='r')
        assert np.isfinite(weight).all() and weight.min()>=0
        if require_single_source:assert weight.max()<=1
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
            selection_path=warp_root/'preflight/sources.json'
            assert receipt['sources']==read(selection_path)['sources'][str(cid)][:config['count']]
            bind(selection_path)
        single_verified=False
        if receipt.get('texture_source_count_per_pixel')==1:
            from transfer_luster_hybrid import merge_details
            original_sampling=(warp_root/'warps').resolve()==(root/'warps').resolve() and (warp_root/'geometry').resolve()==(root/'geometry').resolve()
            area_sampling=warp_root.resolve()==(root/'variants/area4').resolve()
            cubic_sampling=warp_root.resolve() in [(root/'variants/cubic4').resolve(),(root/'variants/cubic75').resolve()]
            assert original_sampling or area_sampling or cubic_sampling, 'Unreviewed single-source sampling variant'
            if cubic_sampling:
                assert (warp_root/'geometry').resolve()==(root/'geometry').resolve()
                for source in receipt['sources']:
                    cd=warp_root/'warps'/f'cam{cid:03d}'/f'src{source:03d}'
                    cr=read(cd/'receipt.json')
                    assert cr['resampling']=='native_keys_cubic_clamped'
                    kernel75=warp_root.resolve()==(root/'variants/cubic75').resolve()
                    assert cr['cubic_code_sha256']==inventory['cubic75_luster_hybrid.py' if kernel75 else 'cubic_luster_hybrid.py']
                    assert cr['tap_count']==16 and cr['cubic_parameter']==(-.75 if kernel75 else -.5)
                    assert cr['lookup_shift'] is False
                    if deep:
                        for field,digest in cr['array_hashes'].items():
                            assert sha(cd/f'{field}.npy')==digest
                            bind(cd/f'{field}.npy')
            if area_sampling:
                for source in receipt['sources']:
                    ar=read(warp_root/'warps'/f'cam{cid:03d}'/f'src{source:03d}'/'receipt.json')
                    assert ar['resampling']=='area_footprint_excess_minification'
                    assert ar['area_code_sha256']==inventory['antialias_luster_hybrid.py']
                    assert ar['quadrature']==4
            assert receipt.get('lookup_shift') is False
            first=warp_root/'warps'/f'cam{cid:03d}'/f"src{receipt['sources'][0]:03d}"
            x0,y0,x1,y1=read(first/'receipt.json')['box'];sl=np.s_[y0:y1,x0:x1]
            local_labels=np.asarray(labels[sl])
            if config['mode']=='missing_bands':
                local_labels=np.load(candidate/f'cam{cid:03d}_construction_sources.npy')
                np.testing.assert_array_equal(labels[sl],np.where(weight[sl]>0,local_labels,-1))
                assert set(map(int,np.unique(local_labels[local_labels>=0])))<=allowed
            texture=np.asarray(baseline[sl]).copy();model_texture=texture.copy()
            for source in receipt['sources']:
                d=warp_root/'warps'/f'cam{cid:03d}'/f'src{source:03d}'
                selected=local_labels==source
                if not selected.any():continue
                assert np.all(np.load(d/'valid.npy',mmap_mode='r')[selected]>0)
                assert np.all(np.load(d/'score.npy',mmap_mode='r')[selected]<=config['max_score'])
                photo=np.load(d/'rgb.npy',mmap_mode='r');texture[selected]=photo[selected]
                if config['mode'] in ['single_source_model_residual','single_source_view_illumination']:
                    model=np.load(root/'source_model_warps'/f'cam{cid:03d}'/f'src{source:03d}'/'rgb.npy',mmap_mode='r');model_texture[selected]=model[selected]
                for field,limit,lower in [('alpha',config['alpha'],True),('resolution_ratio',config['min_resolution'],True),('depth_error',config['depth_error'],False),('roundtrip',config['roundtrip'],False),('uncertainty',config['uncertainty'],False)]:
                    values=np.load(d/f'{field}.npy',mmap_mode='r')[selected]
                    assert np.all(values>=limit) if lower else np.all(values<=limit)
            mode='detail' if config['mode'] in ['single_source_model_residual','missing_bands','single_source_view_illumination','single_source_seam'] else config['mode']
            rebuilt,rebuilt_weight=merge_details(np.asarray(baseline[sl]),texture,local_labels,(local_labels>=0).astype(np.float32),mode,config['sigma'],config['gain'])
            if config['mode']=='single_source_seam':
                from seam_single_luster_hybrid import merge
                parent=root/'candidates'/config['parent']
                np.testing.assert_array_equal(labels,np.load(parent/f'cam{cid:03d}_sources.npy',mmap_mode='r'))
                records=[]
                for source in receipt['sources']:
                    directory=warp_root/'warps'/f'cam{cid:03d}'/f'src{source:03d}'
                    wr=read(directory/'receipt.json')
                    records.append((source,{field:np.load(directory/f'{field}.npy',mmap_mode='r') for field in wr['arrays']}))
                rebuilt,rebuilt_weight=merge(np.asarray(baseline[sl]),records,local_labels,config)
            if config['mode']=='single_source_view_illumination':
                from view_illumination_luster_hybrid import merge
                parent=root/'candidates'/config['parent']
                np.testing.assert_array_equal(labels,np.load(parent/f'cam{cid:03d}_sources.npy',mmap_mode='r'))
                rebuilt,rebuilt_weight=merge(np.asarray(baseline[sl]),texture,model_texture,local_labels,config)
            if config['mode']=='single_source_model_residual':
                from residual_luster_hybrid import merge_residual
                rebuilt_weight*=config['residual_gain']
                rebuilt=merge_residual(np.asarray(baseline[sl]),texture-model_texture,local_labels,rebuilt_weight,config['sigma'])
            if config['mode']=='missing_bands':
                from missing_bands_luster_hybrid import merge_missing
                rebuilt,rebuilt_weight=merge_missing(np.asarray(baseline[sl]),texture,local_labels,config)
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
         single_source_sampling='native_keys_cubic_clamped' if warp_root.resolve() in [(root/'variants/cubic4').resolve(),(root/'variants/cubic75').resolve()] else 'native_pixel_footprint' if warp_root.resolve()==(root/'variants/area4').resolve() else 'point' if (warp_root/'warps').resolve()==(root/'warps').resolve() else 'other_variant',
         uses_source_model_residual=config.get('mode') in ['source_model_residual','single_source_model_residual'],
         uses_frozen_source_view_illumination=config.get('mode')=='single_source_view_illumination',
         uses_adaptive_source_seam_confidence=config.get('mode')=='single_source_seam' and config.get('adaptive_seam',False),
         code=inventory)
    write(candidate/'provenance.json',report)
    print(name,': source exclusion, finite outputs and exact fallback verified; deep=',deep,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--warp-root',type=Path)
    p.add_argument('--deep',action='store_true');p.add_argument('--require-single-source',action='store_true');a=p.parse_args()
    freeze(a.root,a.name,a.warp_root or a.root,a.deep,a.require_single_source)
