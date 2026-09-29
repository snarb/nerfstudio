"""Freeze real RGB with all three DEC5 held-out cameras, without teacher targets."""
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from joint_temporal_texture import SOURCE, CALIBRATION, CAMERA_FIELDS, read, sha, exr, display
from render_patchmatch_camera_path import normalize_frame
import numpy as np
from PIL import Image


def main():
    root=Path('/home/brans/lookcloser_artifacts/blur_ablation_fresh')
    out=root/'real';out.mkdir(parents=True,exist_ok=True)
    source=Path('/mnt/data/dec5_000973_mesh_distillation_v1/real')
    meta=read(source/'transforms.json');meta=deepcopy(meta)
    (out/'images').mkdir(exist_ok=True)
    for row in meta['frames']:
        p=out/row['file_path']
        if not p.exists():p.symlink_to(source/row['file_path'])
    calibration=read(CALIBRATION);by_name={r['physical_camera']:r for r in calibration['frames']}
    originals=read(SOURCE/'000973/transforms.json')
    geometry=read('/mnt/data/lookcloser_dec5_5a3_surface_repair/diagnostics/000973/full_block_tsdf/mesh.json')
    gain=read('/mnt/data/lookcloser_dec5_5a3_joint_texture/exposure.json')['fixed_exposure_gain']
    present={r['physical_camera'] for r in meta['frames']}
    for original in originals['frames']:
        if 'eval' not in original['file_path'] or original['physical_camera'] in present:continue
        row=deepcopy(original)
        for key in CAMERA_FIELDS:row[key]=deepcopy(by_name[row['physical_camera']][key])
        row=normalize_frame(row,calibration,geometry)
        rel=f'images/eval_{len(meta["frames"]):04d}.png'
        rgb=display(exr(SOURCE/'000973'/original['file_path']),gain)
        Image.fromarray(np.rint(rgb*255).clip(0,255).astype('uint8')).save(out/rel)
        row['file_path']=rel;row.pop('depth_file_path',None);row.pop('mask_path',None)
        meta['frames'].append(row);meta['val_filenames'].append(rel);meta['test_filenames'].append(rel)
    # Full-scene bounds in the already frozen coordinate frame. Actor AABB is
    # deliberately not used to discard the real room in the main comparison.
    meta['blur_aabb']=[[-1.5,-1.5,-1.5],[1.5,1.5,1.5]]
    (out/'transforms.json').write_text(json.dumps(meta,indent=2)+'\n')
    (out/'input_hashes.json').write_text(json.dumps({r['file_path']:sha(out/r['file_path']) for r in meta['frames']},indent=2)+'\n')
    print('real',len(meta['train_filenames']),len(meta['val_filenames']),out)
    source=Path('/mnt/data/dec5_lookcloser_recovery_v2/synthetic_single')
    out=root/'synthetic_single';out.mkdir(exist_ok=True)
    for p in source.iterdir():
        if p.is_dir() and not (out/p.name).exists():(out/p.name).symlink_to(p)
    meta=read(source/'transforms.json');meta['blur_aabb']=meta['distillation']['actor_bounds']
    (out/'transforms.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('synthetic',out)


if __name__=='__main__':main()
