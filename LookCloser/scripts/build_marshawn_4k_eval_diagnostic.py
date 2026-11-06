#!/usr/bin/env python3
"""Isolated CPU EVAL RGB/coverage export; fixed V1 and TRAIN remain untouched."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def eval_gain(fit,camera):
    if camera not in {'DC','HC','KA'}or fit['uses_eval_RGB']is not False or fit['k']!=4:raise ValueError('Frozen EVAL-only gains required')
    if any(r['camera']==camera for r in fit['camera_rows']):raise ValueError('EVAL camera found in fitted TRAIN rows')
    return float(fit['eval_predictions'][camera]['ev'])
def photo(linear,row,contract,fit):
    import numpy as np
    from marshawn_radial_photometry import radial_ev
    from prepare_marshawn_color_variants import srgb
    if contract['global_base_gain']!=4 or contract['radial_and_scalar_baked']is not True:raise ValueError('Wrong frozen photometry')
    base=(linear*contract['global_base_gain']).astype(np.float32)
    field=radial_ev(row,fit['k'])-fit['radial_global_center_EV']+eval_gain(fit,row['physical_camera'])
    return srgb((base.astype(np.float64)*np.exp2(field[...,None])).astype(np.float32))
def coverage(row):
    from marshawn_actor_coordinates import maps
    mx,my=maps(row);return (mx>=0)&(mx<=8191)&(my>=0)&(my<=4319)
def verify_rays(row,high):
    import numpy as np
    uv=np.array([[.5,.5],[row['cx'],row['cy']],[row['w']-.5,row['h']-.5]],np.float64)
    ray=(uv-[row['cx'],row['cy']])/[row['fl_x'],row['fl_y']]
    out=(uv*2-[high['cx'],high['cy']])/[high['fl_x'],high['fl_y']]
    if not np.array_equal(ray,out):raise ValueError('Scaled continuous camera rays changed')

def one(payload):
    import cv2,numpy as np
    from PIL import Image
    from prepare_marshawn_color_variants import linear_hd,vendor
    from build_marshawn_4k_pilots import scaled_camera,array_sha
    row,contract,fit,data,output=payload;data=Path(data);output=Path(output);camera=row['physical_camera'];started=time.time();cv2.setNumThreads(1)
    pq=vendor();low,raw=linear_hd(row,pq);rgb=photo(low,row,contract,fit);del low
    expected=np.asarray(Image.open(data/row['file_path']).convert('RGB'));valid=coverage(row);expected_valid=np.asarray(Image.open(data/row['mask_path']).convert('L'))>0
    if not np.array_equal(rgb,expected):raise ValueError('FHD RGB byte replay failed '+camera)
    if not np.array_equal(valid,expected_valid):raise ValueError('FHD validity replay failed '+camera)
    write(output/'replay'/f'{camera}.json',dict(camera=camera,FHD_RGB_uint8_byte_exact=True,FHD_valid_exact=True,replay_RGB_array_sha256=array_sha(rgb),replay_valid_array_sha256=array_sha(valid),original_RGB_file_sha256=sha(data/row['file_path']),original_valid_file_sha256=sha(data/row['mask_path'])))
    del rgb,valid,expected,expected_valid
    high=scaled_camera(row,2);verify_rays(row,high);linear,raw2=linear_hd(high,pq)
    if raw2['raw_sha256']!=raw['raw_sha256']:raise ValueError('RAW changed during export')
    rgb=photo(linear,high,contract,fit);del linear;valid=coverage(high)
    rp=output/'images'/f'{camera}.png';vp=output/'valid'/f'{camera}.png';Image.fromarray(rgb).save(rp);Image.fromarray(valid.astype(np.uint8)*255).save(vp)
    if not np.array_equal(np.asarray(Image.open(rp)),rgb):raise ValueError('PNG roundtrip mismatch')
    high['file_path']='images/'+camera+'.png';high['mask_path']='valid/'+camera+'.png'
    receipt=dict(camera=camera,raw=raw,scene_frame=1660,local_RAW_frame=408,source_frame=row,diagnostic_frame=high,scalar_EV=eval_gain(fit,camera),RGB_sha256=sha(rp),valid_sha256=sha(vp),RGB_shape=list(rgb.shape),valid_pixels=int(valid.sum()),FHD_replay_exact=True,normalized_continuous_rays_equal=True,elapsed_seconds=time.time()-started,training_eligible=False,semantic_masks_generated=False)
    write(output/'receipts'/f'{camera}.json',receipt);return receipt

def run(spec_path):
    spec=read(spec_path);output=Path(spec['output']);started=time.time()
    if spec['workers']not in (1,2)or spec['cameras']!=['DC','HC','KA']:raise ValueError('Scope exceeds released EVAL3')
    release=read(output/'root_release.json')
    if release.get('authorized_CPU_export')is not True or release.get('spec_sha256')!=sha(spec_path):raise ValueError('Exact CPU release required')
    for p,digest in spec['source_hashes'].items():
        if sha(p)!=digest:raise ValueError('Source changed '+p)
    if (output/'DIAGNOSTIC_ONLY.json').exists():raise ValueError('Generation already exported')
    import build_marshawn_4k_pilots as base
    data=Path(spec['data']);meta,ready,manifest,contract,fit,binding,evals=base.source_contract(data)
    original={str(data/rel):h for e in evals for rel,h in e['files'].items()}
    original.update({str(data/n):sha(data/n)for n in ('transforms.json','ready.json','box.json')})
    if len(evals)!=3:raise ValueError('Unexpected EVAL membership')
    rows=[next(r for r in meta['frames']if r['physical_camera']==c)for c in spec['cameras']]
    if any(r['file_path']in meta['train_filenames']for r in rows):raise ValueError('EVAL appeared in TRAIN')
    write(output/'original_preservation_before.json',original)
    for name in ['images','valid','receipts','replay']:(output/name).mkdir(exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=spec['workers'])as pool:
        receipts=list(pool.map(one,[(r,contract,fit,str(data),str(output))for r in rows]))
    for p,digest in {**original,**spec['source_hashes']}.items():
        if sha(p)!=digest:raise ValueError('Source changed during export '+p)
    descriptor=dict(kind='SUPPLEMENTARY_EVAL4K_VISUAL_ONLY',training_eligible=False,primary_metric_protocol='unchanged V1 FHD',frame=1660,frames=[r['diagnostic_frame']for r in receipts],source_descriptor_sha256=original[str(data/'transforms.json')],source_box_sha256=original[str(data/'box.json')],world_to_box=ready['world_to_box'],visualization_only_crops={e['frame']['physical_camera']:{a:[2*x for x in b]for a,b in e['crops'].items()}for e in evals})
    write(output/'render_views_4k.json',descriptor);write(output/'original_preservation_after.json',original)
    write(output/'DIAGNOSTIC_ONLY.json',dict(status='RGB_and_calibration_coverage_only',CPU_only=True,workers=spec['workers'],source_spec_sha256=sha(spec_path),source_hashes=spec['source_hashes'],receipts={r['camera']:sha(output/'receipts'/(r['camera']+'.json'))for r in receipts},render_views_sha256=sha(output/'render_views_4k.json'),original9EVAL_and_descriptors_unchanged=True,FHD_replays_byte_exact=True,training_eligible=False,primary_V1_unchanged=True,semantic_masks_generated=False,metrics_generated=False,model_rendered=False,elapsed_seconds=time.time()-started,manual_review_complete=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);a=p.parse_args()
    try:run(a.spec)
    except BaseException as exc:
        out=Path(read(a.spec)['output']);write(out/'failed.json',dict(error=repr(exc),time=time.time()));raise
