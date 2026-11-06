#!/usr/bin/env python3
"""Reviewed scope extension of the exact four-pilot CPU builder to TRAIN63."""
import argparse, fcntl, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import build_marshawn_4k_pilots as base

HELPERS=['build_marshawn_4k_pilots.py','prepare_marshawn_color_variants.py','marshawn_actor_coordinates.py','marshawn_actor_confidence.py','marshawn_radial_photometry.py']


def validate_review(review_path, data):
    review=json.loads(review_path.read_text());pilot=review_path.parent
    if review.get('reviewer')!='root' or not review.get('authorized_next','').startswith('CPU export all63 TRAIN using exact same builder'):
        raise ValueError('Explicit root review authorizing only full63 CPU export required')
    if base.sha(pilot/'complete.json')!=review.get('complete_sha256'):raise ValueError('Reviewed pilot completion changed')
    plan=json.loads((pilot/'plan.json').read_text())
    for path,h in plan['source_bindings'].items():
        p=Path(path);actual=Path(__file__).with_name(p.name) if p.name in HELPERS else p
        if base.sha(actual)!=h:raise ValueError('Reviewed helper/source changed: '+str(actual))
    if str(data/'ready.json') not in plan['source_bindings']:raise ValueError('Different source dataset')
    return review


def export(data,output,review_path):
    review=validate_review(review_path,data);meta=json.loads((data/'transforms.json').read_text());ready=json.loads((data/'ready.json').read_text())
    names=tuple(r['physical_camera'] for r in meta['frames'] if r['file_path'] in meta['train_filenames'])
    if len(names)!=63 or len(set(names))!=63 or set(names)&{'DC','HC','KA'}:raise ValueError('Expected63 unique TRAIN cameras')
    output.mkdir(parents=True,exist_ok=True)
    generation={'parent_review':str(review_path),'parent_review_sha256':base.sha(review_path),'scope':'TRAIN63 CPU source export only; no maps or training',
                'wrapper_sha256':base.sha(__file__),'exact_parent_builder_sha256':base.sha(Path(base.__file__)),'cameras':list(names),'cpu_workers':2,
                'inherited_v1_semantic_warning':'MC grey-foot bag inclusion remains inherited source-mask error; masks are not geometric GT and were not silently corrected.'}
    gp=output/'generation.json'
    if gp.exists() and json.loads(gp.read_text())!=generation:raise ValueError('Immutable export generation changed')
    base.atomic(gp,generation)
    # Only authorized camera scope changes. RGB, masks, replay, overlays, CPU
    # concurrency and source validation execute the byte-identical reviewed code.
    base.PILOTS=names
    base.run(data,output,2)
    shutil.copy2(output/'pilot_transforms.json',output/'transforms.json')
    records=[]
    for row in meta['frames']:
        camera=row['physical_camera'];record=dict(next(r for r in ready['records'] if r['camera']==camera))
        record.update(semantic=str(output/'semantics'/(camera+'.npz')),semantic_sha256=base.sha(output/'semantics'/(camera+'.npz')),rgb_sha256=base.sha(output/row['file_path']))
        if camera in names:
            with np.load(output/'semantics'/(camera+'.npz'),allow_pickle=False) as z:
                record['crops']={actor:base.bounds(z[actor],64) for actor in ['black','grey']}
            record['source_replay_receipt_sha256']=base.sha(output/'replay'/(camera+'.json'))
        records.append(record)
    source_metadata={k:ready[k] for k in ['frame','train','validation','aabb','world_to_box','source_confidence_sha256']}
    source_metadata.update(records=records,transforms_sha256=base.sha(output/'transforms.json'),box_sha256=base.sha(output/'box.json'),
                           generation_sha256=base.sha(gp),frequency_policy='UNQUALIFIED: new TRAIN63 maps required; do not copy FHD maps',training_eligible=False,
                           inherited_v1_semantic_warning=generation['inherited_v1_semantic_warning'])
    base.atomic(output/'source_metadata.json',source_metadata)
    result=json.loads((output/'complete.json').read_text());result.update(status='TRAIN63_sources_complete_maps_unqualified',training_eligible=False,
               source_metadata_sha256=base.sha(output/'source_metadata.json'),generation_sha256=base.sha(gp),parent_review_sha256=base.sha(review_path),
               inherited_v1_semantic_warning=generation['inherited_v1_semantic_warning'])
    base.atomic(output/'complete.json',result)
    if (output/'ready.json').exists() or (output/'lookcloser_frequencies').exists():raise ValueError('Source-only export must not create READY or map directory')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','launch','status']);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--review',type=Path,required=True);a=p.parse_args()
    if a.action=='status':print((a.output/'progress.json').read_text());return
    validate_review(a.review,a.data);a.output.mkdir(parents=True,exist_ok=True)
    if a.action=='launch':
        with (a.output/'launcher.lock').open('a') as held:
            fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
            old=json.loads((a.output/'launcher.json').read_text()) if (a.output/'launcher.json').exists() else {}
            if base.alive(old.get('process')):print('Builder already alive');return
            with (a.output/'builder.log').open('ab') as log:
                argv=[sys.executable,str(Path(__file__).resolve()),'run','--data',str(a.data),'--output',str(a.output),'--review',str(a.review)]
                child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            base.atomic(a.output/'launcher.json',{'process':base.identity(child.pid),'time':time.time(),'cpu_workers':2});print(json.dumps({'pid':child.pid}));return
    try:export(a.data,a.output,a.review)
    except Exception as exc:
        base.atomic(a.output/'progress.json',{'pid':os.getpid(),'time':time.time(),'status':'failed','error':repr(exc)});raise


if __name__=='__main__':main()
