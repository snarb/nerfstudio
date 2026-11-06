"""Read-only TRAIN positive-support counts, keeping UNKNOWN separate from support.

Neither low support nor a source-mask intersection certifies physical geometry.
No gate is changed, and no field or CUDA context is loaded.
"""
from pathlib import Path
import argparse,json,hashlib,time
import numpy as np
from mask_visual_hull_aabb import MaskCamera,classify_cells
from build_marshawn_semantic_resolution import cell_bounds


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    b=a.base;data=b.parent/'data'
    source_path=b/'occlusion_v2/W64/support_manifest.json';source={r['camera']:r for r in json.loads(source_path.read_text())['records']}
    ready=json.loads((data/'ready.json').read_text());records={r['camera']:dict(r) for r in ready['records']}
    for r in json.loads((b/'corrected_v2/mask_corrections_final.json').read_text())['records']:records[r['camera']].update(semantic=r['corrected_archive'],semantic_sha256=r['corrected_archive_sha256'])
    tf=json.loads((data/'transforms.json').read_text());frames={r['physical_camera']:r for r in tf['frames'] if r['file_path'] in tf['train_filenames']};assert len(frames)==63 and set(source)==set(frames) and not(set(frames)&{'DC','HC','KA'})
    grid_path=b/'occlusion_v2/W64/semantic_grid.npz'
    with np.load(grid_path) as z:gate0=z['semantic_gate0'].copy();pad1=z['semantic_gate_pad1'].copy();bounds=z['aabb'].copy()
    n=gate0.shape[0];assert gate0.shape==(128,128,128)
    material=json.loads((b/'mat_geometry_TRAIN/qualified_material_points_v2.json').read_text())
    anchors=[dict(name='toe_upper',xyz_floor=[-.11769208960984658,.15127923559030565,.014567836967476211],label='independent_TRAIN_toe_upper')]+[dict(name='sfm_'+str(r['source_point_id']),xyz_floor=r['xyz_floor'],label=r['label']) for r in material['records'] if r['qualified_mat_material'] or r['label']=='black_shoe_sole_material']
    for r in anchors:
        idx=np.floor((np.asarray(r['xyz_floor'])-bounds[0])/(bounds[1]-bounds[0])*n).astype(int);assert ((idx>=0)&(idx<n)).all();r['index']=idx.tolist();r['flat_index']=int(np.ravel_multi_index(idx,gate0.shape))
    ids=np.unique(np.r_[np.flatnonzero(pad1),[r['flat_index'] for r in anchors]])
    keys=['W64_overlap','W64_overlap_no_unknown_footprint','nominal_actor_overlap','nominal_actor_overlap_no_unknown_footprint','unknown_or_invalid_views','clear_background_views']
    counts={k:np.zeros(len(ids),np.uint8) for k in keys};anchor_rows={r['name']:[] for r in anchors};anchor_slots={r['name']:int(np.searchsorted(ids,r['flat_index'])) for r in anchors};sources=[];started=time.monotonic()
    for camera in sorted(frames):
        f=frames[camera];s=source[camera];record=records[camera];assert sha(s['archive'])==s['archive_sha256'] and sha(record['semantic'])==record['semantic_sha256']
        with np.load(s['archive']) as z:support=z['support'].copy();unknown=z['unknown'].copy();valid=z['valid'].copy()
        with np.load(record['semantic']) as z:nominal=(z['black']|z['grey']).copy()
        args=(camera,np.asarray(f['transform_matrix']),f['fl_x'],f['fl_y'],f['cx'],f['cy'])
        wide=MaskCamera(*args,support,unknown,valid).prepare(0);narrow=MaskCamera(*args,nominal,unknown,valid).prepare(0)
        single={k:np.zeros(len(ids),np.uint8) for k in keys}
        for start in range(0,len(ids),32768):
            sl=slice(start,start+32768);lo,hi=cell_bounds(ids[sl],n,bounds);fg,_,bg,un=classify_cells([wide],lo,hi);nom,_,_,_=classify_cells([narrow],lo,hi)
            values=[fg,(fg>0)&(un==0),nom,(nom>0)&(un==0),un,bg]
            for k,v in zip(keys,values):single[k][sl]=v
        for k in keys:counts[k]+=single[k]
        for r in anchors:anchor_rows[r['name']].append(dict(camera=camera,**{k:int(single[k][anchor_slots[r['name']]]) for k in keys}))
        sources.append(dict(camera=camera,support_archive=s['archive'],support_sha256=s['archive_sha256'],semantic=record['semantic'],semantic_sha256=record['semantic_sha256']))
        print(json.dumps(dict(camera=camera,completed=len(sources),seconds=time.monotonic()-started)),flush=True)
    selected0=gate0.reshape(-1)[ids];selected1=pad1.reshape(-1)[ids];lo,hi=cell_bounds(ids,n,bounds)
    groups={'gate0':selected0,'pad1':selected1,'pad1_only':selected1&~selected0,'pad1_nominal_floor_band_NOT_surface_label':selected1&(lo[:,2]<.01)&(hi[:,2]>-.01),'pad1_z_above_point1_NOT_body_label':selected1&(lo[:,2]>.1)}
    histograms={name:dict(cells=int(sel.sum()),counts={k:np.bincount(v[sel],minlength=64).tolist() for k,v in counts.items()}) for name,sel in groups.items()}
    for r in anchors:
        slot=anchor_slots[r['name']];r.update(gate0=bool(selected0[slot]),pad1=bool(selected1[slot]),counts={k:int(v[slot]) for k,v in counts.items()},source_view_votes=anchor_rows[r['name']])
    np.savez_compressed(a.output/'counts.npz',flat_indices=ids,**counts,aabb=bounds,resolution=np.array([n]*3))
    result=dict(kind='TRAIN_POSITIVE_SUPPORT_DIAGNOSTIC_NO_CARVING',resolution=n,grid_path=str(grid_path),grid_sha256=sha(grid_path),source_manifest_sha256=sha(source_path),counts_sha256=sha(a.output/'counts.npz'),histograms=histograms,anchors=anchors,sources=sources,seconds=time.monotonic()-started,definitions={'overlap':'At least one valid actor-support pixel in outward rectangle of projected8corners, including partial-raster rectangles. Weak2D silhouette support, not3D occupancy.','no_unknown_footprint':'Same positive overlap but entire rectangle is valid/in-raster and has no UNKNOWN pixels. Conservative mat envelopes can also cover real actor pixels, so this is not a ground-truth visibility classifier.','nominal_actor':'Existing corrected black|grey TRAIN labels; remaining bag-label boundary errors are preserved and disclosed.','W64':'Exact frozen64-native-pixel support; padding can support nonactor volume.','unknown':'Independent visibility/invalid/out-of-raster status, never converted to a positive or negative material label.'},no_eval_input=True,no_field_or_GPU_loaded=True,no_gate_modified=True,limitations=['Low or zero support is not ground-truth empty; hidden real geometry and label errors may lack positive evidence.','Existing UNKNOWN coverage remains partial.','Nominal floor height band is a coordinate region, not a fitted floor plane.','Ray coverage under hypothetical support thresholds is separate from actor-surface retention.'])
    (a.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({name:{k:int(v[sel].min()) if sel.any() else None for k,v in counts.items()} for name,sel in groups.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
