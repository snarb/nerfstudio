"""Strict frame-local ALL66 image-derived admission; no learned occupancy reuse."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.ndimage import maximum_filter
from marshawn_semantic_cell_admission import WholeCellAdmission

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def load_support(manifest_path,expected_sha,data_dir,bounds,resolution,identity=False):
    path=Path(manifest_path);data=Path(data_dir)
    if sha(path)!=expected_sha:raise ValueError('Video admission manifest changed')
    meta=json.loads(path.read_text())
    if meta['kind']!='FRAME_LOCAL_ALL66_W64_PAD1_WHOLE_CELL_V1' or meta['train_camera_count']!=66:raise ValueError('Wrong video admission source')
    for name,digest in meta['files_sha256'].items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or sha(path.parent/rel)!=digest:raise ValueError('Video artifact changed: '+name)
    record=json.loads((path.parent/'grid_receipt.json').read_text());ready=json.loads((data/'ready.json').read_text());tf=json.loads((data/'transforms.json').read_text())
    names=sorted(f['physical_camera'] for f in tf['frames'] if f['file_path'] in tf['train_filenames'])
    if len(set(names))!=66 or names!=sorted(record['train_cameras']):raise ValueError('ALL66 membership changed')
    if record['frame']!=ready['frame'] or record['source_ready_sha256']!=sha(data/'ready.json') or record['transforms_sha256']!=sha(data/'transforms.json'):raise ValueError('Frame-local source binding changed')
    if record['native_pixel_support_width']!=64 or record['no_learned_state_loaded'] is not True:raise ValueError('Wrong support derivation')
    if record['grid_sha256']!=sha(path.parent/'semantic_grid.npz'):raise ValueError('Grid changed')
    if ready['frame']==1660:
        inputs=data/'admission_inputs/manifest.json';prepared=json.loads((data/'prepared_ready.json').read_text())
        if record.get('admission_inputs_manifest_sha256')!=sha(inputs) or prepared.get('admission_inputs_manifest_sha256')!=sha(inputs):raise ValueError('Portable image-admission INPUT binding changed')
        source=json.loads(inputs.read_text())
        if source['frame']!=1660 or source['train_camera_count']!=66 or sorted(r['camera'] for r in source['records'])!=names:raise ValueError('Portable admission camera membership changed')
        for r in source['records']:
            relative=Path(r['archive'])
            if relative.is_absolute() or '..' in relative.parts or sha(data/relative)!=r['archive_sha256']:raise ValueError('Portable image-admission INPUT changed: '+r['camera'])

    with np.load(path.parent/'semantic_grid.npz',allow_pickle=False) as z:
        source=z['semantic_gate_pad1'].copy();base=z['semantic_gate0'].copy();box=z['aabb'].copy();transform=z['world_to_box'].copy()
    if base.dtype!=np.bool_ or source.dtype!=np.bool_ or source.shape!=(128,128,128) or not np.array_equal(source,maximum_filter(base,size=3,mode='constant',cval=False)):raise ValueError('Exactly one fresh halo required')
    np.testing.assert_allclose(box,ready['aabb'],rtol=0,atol=1e-12);np.testing.assert_allclose(transform,ready['world_to_box'],rtol=0,atol=1e-12);np.testing.assert_allclose(bounds,box,rtol=0,atol=2e-7)
    if tuple(resolution)!=(128,128,128):raise ValueError('128 cubed required')
    mapped=WholeCellAdmission(source,box).map_target_grid(resolution,bounds)
    digest=hashlib.sha256(mapped.tobytes()).hexdigest()
    if digest!=meta['mapped_bool_sha256']:raise ValueError('Whole-cell cover changed')
    if identity:mapped=np.ones_like(mapped)
    return mapped,dict(manifest_sha256=expected_sha,source_grid_sha256=record['grid_sha256'],mapped_bool_sha256=hashlib.sha256(mapped.tobytes()).hexdigest(),frame=ready['frame'],train_camera_count=66,monitoring='DC/HC/KA in-sample',identity_control=identity)
