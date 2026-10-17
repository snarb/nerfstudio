"""Audit a complete candidate and bind it to its actual warp variant and outputs."""
import argparse
from pathlib import Path
import time
import numpy as np
from dependency_audit_luster_hybrid import verify_dependencies
from luster_hybrid_common import (DIGEST,TARGETS,allowed_sources,geometry,native_rows,read,sha,source_path,write)


def verify_plate_merge(base,records,parent_labels,vetos,labels,pred,weight,config):
    """Reconstruct the plate-rejected merge without calling its generator.

    The parent supplies labels, while saved native-footprint vetoes remove both
    selected pixels and neighboring filter support. No new source can appear.
    """
    from supported_luster_hybrid import geometric_support
    from seam_single_luster_hybrid import merge
    expected_labels=np.asarray(parent_labels).copy();filtered=[]
    for source,arrays in records:
        veto=np.asarray(vetos[source]);assert veto.dtype==np.bool_
        assert veto.shape==expected_labels.shape
        arrays={**arrays,'valid':(arrays['valid']>0)&~veto}
        expected_labels[(expected_labels==source)&~geometric_support(arrays,config)]=-1
        filtered.append((source,arrays))
    rebuilt,rebuilt_weight=merge(base,filtered,expected_labels,config)
    expected_labels[rebuilt_weight==0]=-1
    np.testing.assert_array_equal(labels,expected_labels)
    np.testing.assert_array_equal(pred,rebuilt)
    np.testing.assert_array_equal(weight,rebuilt_weight)
    return rebuilt,rebuilt_weight


def freeze(root,name,warp_root,deep=False,require_single_source=False):
    rows=native_rows();candidate=root/'candidates'/name
    manifest=read(root/'preflight/manifest.json');config=read(candidate/'config.json')
    inputs={};checks=[];seen_geometry=set();dependency_seen=set()
    state_hash=read(root/'geometry/cam007/receipt.json')['state_hash']
    inventory={p.name:sha(p) for p in sorted((root/'code/LookCloser/scripts').glob('*luster_hybrid*.py'))}
    def bind(path):
        path=Path(path).resolve()
        if str(path) not in inputs:inputs[str(path)]=sha(path)
    for cid in TARGETS:
        receipt=read(candidate/f'cam{cid:03d}_receipt.json')
        assert receipt['config']==config
        assert receipt['code_sha256'] in inventory.values(), 'Candidate generator changed since creation'
        verify_dependencies(candidate/f'cam{cid:03d}_receipt.json',inventory,bind,deep,dependency_seen,
                            expected_state_hash=state_hash,forbidden_sources={12,95,150,cid})
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
            mode='detail' if config['mode'] in ['single_source_model_residual','missing_bands','single_source_view_illumination','single_source_seam','single_source_plate_seam'] else config['mode']
            rebuilt,rebuilt_weight=merge_details(np.asarray(baseline[sl]),texture,local_labels,(local_labels>=0).astype(np.float32),mode,config['sigma'],config['gain'])
            if config['mode'] in ['single_source_seam','single_source_plate_seam']:
                from seam_single_luster_hybrid import merge
                parent=root/'candidates'/config['parent']
                parent_labels=np.load(parent/f'cam{cid:03d}_sources.npy',mmap_mode='r')
                if config['mode']=='single_source_seam':np.testing.assert_array_equal(labels,parent_labels)
                records=[]
                for source in receipt['sources']:
                    directory=warp_root/'warps'/f'cam{cid:03d}'/f'src{source:03d}'
                    wr=read(directory/'receipt.json')
                    records.append((source,{field:np.load(directory/f'{field}.npy',mmap_mode='r') for field in wr['arrays']}))
                if config['mode']=='single_source_plate_seam':
                    vetos={}
                    assert config['plate_veto_level'] in ['strict10','strict20']
                    for source in receipt['sources']:
                        vd=root/'exact_plate_projected'/f'cam{cid:03d}'/f'src{source:03d}'
                        vr=read(vd/'receipt.json');assert vr['target']==cid and vr['source']==source
                        assert vr['footprint']=='all_nonzero_cubic75_taps'
                        path=vd/f"{config['plate_veto_level']}.npy"
                        assert sha(path)==vr['hashes'][path.name];bind(path);bind(vd/'receipt.json')
                        vetos[source]=np.load(path,mmap_mode='r')
                    rebuilt,rebuilt_weight=verify_plate_merge(np.asarray(baseline[sl]),records,parent_labels[sl],
                        vetos,local_labels,pred[sl],weight[sl],config)
                    exterior=np.ones(labels.shape,bool);exterior[sl]=False
                    assert (labels[exterior]==-1).all() and (weight[exterior]==0).all()
                else:
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
         uses_adaptive_source_seam_confidence=config.get('mode') in ['single_source_seam','single_source_plate_seam'] and config.get('adaptive_seam',False),
         uses_measured_source_plate_veto=config.get('mode')=='single_source_plate_seam',
         code=inventory)
    write(candidate/'provenance.json',report)
    print(name,': source exclusion, finite outputs and exact fallback verified; deep=',deep,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--warp-root',type=Path)
    p.add_argument('--deep',action='store_true');p.add_argument('--require-single-source',action='store_true');a=p.parse_args()
    freeze(a.root,a.name,a.warp_root or a.root,a.deep,a.require_single_source)
