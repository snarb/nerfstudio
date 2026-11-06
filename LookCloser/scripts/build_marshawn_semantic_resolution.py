"""Chunked TRAIN semantic admission grid; field coordinates and occupancy untouched."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter
from mask_visual_hull_aabb import MaskCamera, classify_cells


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cell_bounds(ids, resolution, bounds):
    """C-order [x,y,z] cell indexing without allocating the complete XYZ grid."""
    ijk=np.column_stack([ids//(resolution*resolution),(ids//resolution)%resolution,ids%resolution])
    pitch=(bounds[1]-bounds[0])/resolution
    lower=bounds[0]+ijk*pitch
    return lower,lower+pitch


def build(a):
    assert a.resolution in [128,256] and a.chunk>0
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(a.source_manifest.read_text());assert manifest['native_pixel_support_width']==64
    tf=json.loads(a.transforms.read_text());box=json.loads(a.box.read_text());bounds=np.asarray(box['aabb'])
    frames={r['physical_camera']:r for r in tf['frames'] if r['file_path'] in tf['train_filenames']}
    sources={r['camera']:r for r in manifest['records']}
    assert len(frames)==63 and set(frames)==set(sources) and not(set(frames)&{'DC','HC','KA'})
    bg=np.zeros(a.resolution**3,np.uint8);started=time.monotonic();receipts=[]
    for camera in sorted(frames):
        row=sources[camera];assert sha(row['archive'])==row['archive_sha256']
        with np.load(row['archive']) as z:
            support=z['support'].copy();unknown=z['unknown'].copy();valid=z['valid'].copy()
        f=frames[camera]
        c=MaskCamera(camera,np.asarray(f['transform_matrix']),f['fl_x'],f['fl_y'],f['cx'],f['cy'],support,unknown,valid).prepare(0)
        classified,new=0,0
        for offset in range(0,len(bg),a.chunk):
            ids=np.arange(offset,min(offset+a.chunk,len(bg)));ids=ids[bg[ids]<2]
            if not len(ids):continue
            lower,upper=cell_bounds(ids,a.resolution,bounds)
            _,_,clear,_=classify_cells([c],lower,upper);before=bg[ids].copy();bg[ids]+=clear.astype(np.uint8)
            classified+=len(ids);new+=int(((before<2)&(bg[ids]>=2)).sum())
        receipts.append(dict(camera=camera,source_archive=row['archive'],source_sha256=row['archive_sha256'],classified_cells=classified,newly_rejected=new))
        status=dict(camera=camera,completed=len(receipts),retained_cells=int((bg<2).sum()),seconds=time.monotonic()-started)
        (a.output/'progress.json').write_text(json.dumps(status)+'\n');print(json.dumps(status),flush=True)
    gate=(bg<2).reshape((a.resolution,)*3)
    # Both halos are computed directly from gate0, never by modifying native occupancy.
    pad1=maximum_filter(gate,size=3,mode='constant',cval=False)
    pad2=maximum_filter(gate,size=5,mode='constant',cval=False)
    np.savez_compressed(a.output/'semantic_grid.npz',semantic_gate0=gate,semantic_gate_pad1=pad1,semantic_gate_pad2=pad2,clear_background_count_capped2=bg.reshape(gate.shape),aabb=bounds,world_to_box=np.asarray(box['world_to_box']),resolution=np.asarray(gate.shape))
    result=dict(kind='CHUNKED_TRAIN_SEMANTIC_RESOLUTION_FACTOR',resolution=a.resolution,source_manifest=str(a.source_manifest),source_manifest_sha256=sha(a.source_manifest),transforms_sha256=sha(a.transforms),box_sha256=sha(a.box),builder_sha256=sha(__file__),geometry_helper_sha256=sha(Path(__file__).with_name('mask_visual_hull_aabb.py')),sources=receipts,counts=dict(gate0=int(gate.sum()),pad1=int(pad1.sum()),pad2=int(pad2.sum()),total=int(gate.size)),cell_extent=((bounds[1]-bounds[0])/a.resolution).tolist(),grid_sha256=sha(a.output/'semantic_grid.npz'),seconds=time.monotonic()-started,no_native_field_or_grid_loaded=True,no_eval_sources_used=True,mask_policy='EXACT frozen W64 occlusion_v2 source arrays; reject only after2whole-cell clear TRAIN rectangles; UNKNOWN/veto preserved',padding='Fresh noncumulative cube halos from gate0. At256 pad1 has half the physical extent of128pad1; optional256pad2 matches128pad1 axis extent.',limitations=['Partial occluder coverage remains partial; finer cells do not certify visibility.','Actor-ray intersection does not establish retained actor surface or physical correctness.'])
    (a.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['source-manifest','transforms','box','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--resolution',type=int,required=True);p.add_argument('--chunk',type=int,default=32768)
    build(p.parse_args())
