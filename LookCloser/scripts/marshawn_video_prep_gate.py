"""Build a fresh ALL66 image-projected admission; never read a model checkpoint."""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
from scipy.ndimage import maximum_filter
from prepare_marshawn_tuning import verify_inventory,sha,write
from mask_visual_hull_aabb import MaskCamera,classify_cells
from build_marshawn_semantic_resolution import cell_bounds
from marshawn_semantic_cell_admission import WholeCellAdmission

def build(data,chunk=32768):
    ready=verify_inventory(data);tf=json.loads((data/'transforms.json').read_text());bounds=np.asarray(ready['aabb']);out=data/'admission';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():
        from marshawn_video_support import load_support
        load_support(out/'manifest.json',sha(out/'manifest.json'),data,bounds,(128,)*3);return
    if ready['train']!=66:raise ValueError('All66 cameras required')
    frames={f['physical_camera']:f for f in tf['frames'] if f['file_path'] in tf['train_filenames']};assert len(frames)==66
    bg=np.zeros(128**3,np.uint8);records=[];started=time.time()
    source_manifest=data/'admission_inputs/manifest.json'
    source_rows={}
    if ready['frame']==1660:
        m=json.loads(source_manifest.read_text());assert m['frame']==1660 and m['train_camera_count']==66
        source_rows={r['camera']:r for r in m['records']}
    for row in ready['records']:
        name=row['camera'];f=frames[name];semantic=data/'semantics'/f'{name}.npz'
        with np.load(semantic,allow_pickle=False) as z:
            support=z['W64']>0;valid=z['valid'].copy();unknown=z['unknown'].copy() if 'unknown' in z else np.zeros_like(valid)
        provenance='frame-local native-mask W64 support +explicit UNKNOWN'
        # Frame1660 reviewed source arrays travel as immutable INPUTS.
        # Source gates and learned occupancy are always recomputed locally.
        if ready['frame']==1660:
            src=source_rows[name];archive=data/src['archive']
            if sha(archive)!=src['archive_sha256']:raise ValueError('Admission INPUT changed '+name)
            with np.load(archive,allow_pickle=False) as z:
                support=z['support'].copy();unknown=z['unknown'].copy();valid=z['valid'].copy()
            provenance=src['provenance']
        c=MaskCamera(name,np.asarray(f['transform_matrix']),f['fl_x'],f['fl_y'],f['cx'],f['cy'],support,unknown,valid).prepare(0)
        for offset in range(0,len(bg),chunk):
            ids=np.arange(offset,min(offset+chunk,len(bg)));ids=ids[bg[ids]<2]
            if not len(ids):continue
            lo,hi=cell_bounds(ids,128,bounds);_,_,clear,_=classify_cells([c],lo,hi);bg[ids]+=clear.astype(np.uint8)
        records.append(dict(camera=name,semantic_sha256=sha(semantic),support_pixels=int(support.sum()),unknown_pixels=int(unknown.sum()),provenance=provenance))
        write(out/'progress.json',dict(frame=ready['frame'],camera=name,completed=len(records),retained_cells=int((bg<2).sum()),seconds=time.time()-started))
    base=(bg<2).reshape((128,)*3);pad=maximum_filter(base,size=3,mode='constant',cval=False)
    np.savez_compressed(out/'semantic_grid.npz',semantic_gate0=base,semantic_gate_pad1=pad,aabb=bounds,world_to_box=np.asarray(ready['world_to_box']))
    mapped=WholeCellAdmission(pad,bounds).map_target_grid((128,)*3,bounds)
    receipt=dict(frame=ready['frame'],train_cameras=sorted(frames),native_pixel_support_width=64,no_learned_state_loaded=True,source_ready_sha256=sha(data/'ready.json'),transforms_sha256=sha(data/'transforms.json'),grid_sha256=sha(out/'semantic_grid.npz'),builder_sha256=sha(__file__),geometry_helper_sha256=sha(Path(__file__).with_name('mask_visual_hull_aabb.py')),admission_inputs_manifest_sha256=sha(source_manifest) if source_manifest.exists() else None,records=records,clear_evidence='two distinct calibrated cameras classify the complete projected cell clear; UNKNOWN never clear',counts=dict(base=int(base.sum()),pad1=int(pad.sum()),total=base.size),seconds=time.time()-started,limitations=['Image-derived conservative admission, not a certificate of physical occupancy','No inherited first-frame sparse-witness containment claims'])
    write(out/'grid_receipt.json',receipt)
    write(out/'manifest.json',dict(kind='FRAME_LOCAL_ALL66_W64_PAD1_WHOLE_CELL_V1',train_camera_count=66,files_sha256={n:sha(out/n) for n in ['grid_receipt.json','semantic_grid.npz']},mapped_bool_sha256=hashlib.sha256(mapped.tobytes()).hexdigest(),no_model_or_learned_grid_reuse=True))
    from marshawn_video_support import load_support
    load_support(out/'manifest.json',sha(out/'manifest.json'),data,bounds,(128,)*3)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--chunk',type=int,default=32768);a=p.parse_args();build(a.data,a.chunk)
