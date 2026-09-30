"""Check source identity, calibration transforms, split and frequency inputs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
import torch
from blur_runtime import ProbeParserConfig,sha,write
from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_text,read_images_text


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--require-frequencies',action='store_true');args=p.parse_args()
    root=args.root;data=root/'data';source=root/'source'
    local=json.loads((source/'local_manifest.json').read_text());remote=json.loads((source/'remote_manifest.json').read_text())
    if local!=remote:raise ValueError('Remote/source manifest mismatch')
    for name,digest in local.items():
        if sha(source/name)!=digest:raise ValueError(f'Source changed: {name}')
    meta=json.loads((data/'transforms.json').read_text());bounds=json.loads((data/'bounds_audit.json').read_text())
    train=set(meta['train_filenames']);val=set(meta['val_filenames'])
    if len(train)!=162 or len(val)!=3 or train&val:raise ValueError('Bad split')
    parser=ProbeParserConfig(data=data,downscale_factor=1,auto_scale_poses=False,center_method='none',orientation_method='none').setup()
    outputs=parser.get_dataparser_outputs('train');ev=parser.get_dataparser_outputs('val')
    if len(outputs.image_filenames)!=162 or len(ev.image_filenames)!=3:raise ValueError('Parser split mismatch')
    np.testing.assert_allclose(outputs.scene_box.aabb.numpy(),meta['blur_aabb'],atol=1e-7)
    cs=read_cameras_text(source/'frame/sparse/text/cameras.txt');ims=read_images_text(source/'frame/sparse/text/images.txt')
    transform=np.array(bounds['normalization']);scale=bounds['scale']
    points=np.array([[1.8,.7,.3],[2.,.9,.5],[2.7,.4,.7],[3.5,.2,1.]])
    points_norm=(points@transform[:3,:3].T+transform[:3,3])*scale
    max_error=0.;receipts=0;derived={}
    archived_originals=None
    if (data/'original_hd_archive.json').exists():
        from archive_luster_originals import verify_archived_originals
        archived_originals=verify_archived_originals(data)
    for row in meta['frames']:
        cid=row['camera_id'];name=Path(row['file_path']).name
        for folder in ['images','original_hd','masks']:
            path=data/folder/name
            if folder=='original_hd' and archived_originals is not None:
                if archived_originals[name]['size']!=[row['w'],row['h']]:raise ValueError('Archived original shape mismatch')
                derived[str(path.relative_to(data))]=archived_originals[name]['sha256']
                continue
            if Image.open(path).size!=(row['w'],row['h']):raise ValueError('Prepared shape mismatch')
            derived[str(path.relative_to(data))]=sha(path)
        im=ims[cid];cam=cs[im.camera_id]
        raw=points@im.qvec2rotmat().T+im.tvec
        fx,fy,cx,cy=cam.params
        uv=(raw[:,:2]/raw[:,2:]*[fx,fy]+[cx,cy])*[row['w']/cam.width,row['h']/cam.height]
        pose=np.array(row['transform_matrix']);xyz=(points_norm-pose[:3,3])@pose[:3,:3]
        projected=np.column_stack([xyz[:,0]/-xyz[:,2]*row['fl_x']+row['cx'],-xyz[:,1]/-xyz[:,2]*row['fl_y']+row['cy']])
        max_error=max(max_error,float(np.abs(uv-projected).max()))
        freq=data/'lookcloser_frequencies'/f'{Path(name).stem}.pt'
        if row['file_path'] in val:
            if freq.exists():raise ValueError('Eval image has a frequency map')
            continue
        receipt=freq.with_suffix('.receipt.json')
        if not receipt.exists():
            if args.require_frequencies:raise ValueError(f'Missing frequency receipt: {name}')
            continue
        record=json.loads(receipt.read_text())
        if record['request']['rgb_sha256']!=derived[row['file_path']] or record['sha256']!=sha(freq):raise ValueError(f'Stale frequency map: {name}')
        array=torch.load(freq,map_location='cpu',weights_only=True)
        expected=((row['h']-8)//8+1,(row['w']-8)//8+1)
        if tuple(array.shape)!=expected or not torch.isfinite(array).all() or not (array>0).all():raise ValueError(f'Invalid map: {name}')
        receipts+=1
    if max_error>1e-3:raise ValueError(f'Calibration projection mismatch {max_error} px')
    if min(r['aabb_hit_fraction'] for r in bounds['coverage'])<.9999:raise ValueError('Foreground rays clipped')
    write(data/'derived_manifest.json',derived)
    write(data/('audit_ready.json' if args.require_frequencies else 'audit_preprocessing.json'),dict(
        source_files=len(local),source_hashes_match=True,train=162,eval=3,maximum_projection_error_pixels=max_error,
        frequency_maps=receipts,minimum_foreground_ray_aabb_fraction=min(r['aabb_hit_fraction'] for r in bounds['coverage']),
        transforms_sha256=sha(data/'transforms.json'),derived_manifest_sha256=sha(data/'derived_manifest.json')))
    print(f'sources={len(local)} split=162/3 projection_error_px={max_error:.8f} maps={receipts}/162')


if __name__=='__main__':main()
