"""Isolated conservative vote-threshold variant; never mutate source datasets."""
import argparse
import hashlib
from pathlib import Path
import shutil
import numpy as np
from scipy.ndimage import maximum_filter
from common import read, atomic, sha


def build(source, output, votes):
    if type(votes) is not int or not 3 <= votes <= 72:
        raise ValueError('Explicit more-conservative threshold in[3,72] required')
    src, out = Path(source).resolve(), Path(output).resolve()
    if src == out or out.exists():
        raise ValueError('Fresh separate variant directory required')
    from mask_visual_hull_aabb import MaskCamera, classify_cells
    from build_marshawn_semantic_resolution import cell_bounds
    from marshawn_semantic_cell_admission import WholeCellAdmission
    out.mkdir(parents=True)
    for name in ('images','masks','linear','parts','semantics','heads','cores','lookcloser_frequencies'):
        if (src/name).exists(): (out/name).symlink_to((src/name).resolve(), target_is_directory=True)
    for name in ('ready.json','transforms.json','box.json','units_parity.json','points3D.txt'):
        if (src/name).is_file(): shutil.copy2(src/name, out/name)
    ready, tf = read(out/'ready.json'), read(out/'transforms.json')
    bounds = np.asarray(ready['aabb'])
    bg = np.zeros(128**3, np.uint8)
    records = []
    for f in tf['frames']:
        if f['file_path'] not in tf['train_filenames']: continue
        name = f['physical_camera']
        with np.load(out/'semantics'/(name+'.npz')) as z:
            fg, valid = z['black']|z['grey'], z['valid']
        with np.load(out/'parts'/(name+'.npz')) as z: unknown = z['unknown_people']
        cam = MaskCamera(name,np.asarray(f['transform_matrix']),f['fl_x'],f['fl_y'],f['cx'],f['cy'],fg,unknown,valid).prepare(64)
        for offset in range(0,len(bg),32768):
            ids = np.arange(offset,min(offset+32768,len(bg))); ids = ids[bg[ids]<votes]
            if not len(ids): continue
            lo, hi = cell_bounds(ids,128,bounds)
            _, _, clear, _ = classify_cells([cam],lo,hi)
            bg[ids] += clear.astype(np.uint8)
        records.append(dict(camera=name,semantic_sha256=sha(out/'semantics'/(name+'.npz'))))
        print('ADMISSION_VOTES',name,int((bg<votes).sum()),flush=True)
    base = (bg<votes).reshape((128,)*3)
    pad = maximum_filter(base,size=3,mode='constant',cval=False)
    mapped = WholeCellAdmission(pad,bounds).map_target_grid((128,)*3,bounds)
    with np.load(src/'admission/grid.npz') as z: parent = z['mapped']
    if np.any(parent & ~mapped): raise ValueError('Conservative variant removed previously admitted cells')
    dest = out/'admission'; dest.mkdir()
    np.savez_compressed(dest/'grid.npz',base=base,pad1=pad,mapped=mapped,aabb=bounds)
    manifest = dict(kind='SEQUENCE_TRAIN_ONLY_W64_WHOLE_CELL_VOTES_V2',background_votes=votes,
        records=records,train_count=len(records),grid_sha256=sha(dest/'grid.npz'),
        ready_sha256=sha(out/'ready.json'),transforms_sha256=sha(out/'transforms.json'),
        mapped_sha256=hashlib.sha256(mapped.tobytes()).hexdigest(),
        parent_manifest_sha256=sha(src/'admission/manifest.json'),
        added_cells=int(np.count_nonzero(mapped & ~parent)),
        clear_evidence=f'{votes} distinct TRAIN whole-cell background votes; original64px support, UNKNOWN handling, pad1 and whole-cell mapping unchanged')
    atomic(dest/'manifest.json',manifest)
    return manifest


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--votes',type=int,required=True)
    a=p.parse_args();print(build(a.source,a.output,a.votes),flush=True)
