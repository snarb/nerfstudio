"""Reuse unchanged verified frames with explicit receipt ancestry, never fake reruns.

Only a same-recipe, same-time, same-camera geometry replacement qualifies.
Replacement pixels still require fresh visual review before video publication.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
from joint_temporal_texture import read,sha,atomic_json
from render_smooth_temporal_mesh_video import verify_request
from finalize_local_mesh_repair import verify_hashes


def assert_texture_equivalent(parent, replacement):
    for k in ['calibration_sha256','profiles_sha256','exposure_sha256','uses_heldout_rgb']:
        if parent[k]!=replacement[k]:raise ValueError(f'Texture input differs: {k}')
    for k,v in parent['recipe'].items():
        if k!='frames' and replacement['recipe'].get(k)!=v:raise ValueError(f'Texture recipe differs: {k}')
    for k,v in parent['script_hashes'].items():
        if replacement['script_hashes'].get(k)!=v:raise ValueError(f'Render dependency differs: {k}')


def link_tree(source,destination):
    destination.mkdir(parents=True,exist_ok=True)
    for p in source.rglob('*'):
        q=destination/p.relative_to(source)
        if p.is_dir():q.mkdir(exist_ok=True)
        elif p.is_file():
            if q.exists():raise FileExistsError(q)
            try:os.link(p,q)
            except OSError:shutil.copyfile(p,q)


def compose(parent, replacement, output):
    a,b=verify_request(parent),verify_request(replacement)
    assert_texture_equivalent(a,b)
    if len(a['ordered_frame_ids'])!=150 or len(b['ordered_frame_ids'])!=1:raise ValueError('Require 150-parent and one replacement time')
    frame=b['ordered_frame_ids'][0]
    old=next(r for r in a['inventory'] if r['frame_id']==frame);new=b['inventory'][0]
    if {k:v for k,v in old.items() if k not in ['mesh','mesh_sha256','metadata','metadata_sha256']} != {
        k:v for k,v in new.items() if k not in ['mesh','mesh_sha256','metadata','metadata_sha256']}:
        raise ValueError('Only geometry may change; time, camera, index and source identity must stay fixed')
    if b['source_rows']!=[r for r in a['source_rows'] if Path(r['source_dataset']).name==frame]:raise ValueError('Source RGB inventory differs')
    if output.exists():raise FileExistsError('Composition uses a new output root')
    # Validate every source before starting publication of the derived workspace.
    records=[]
    for row in a['inventory']:
        source=replacement if row['frame_id']==frame else parent
        directory=source/'frames'/row['frame_id'];receipt=read(directory/'complete.json')
        if receipt['request_sha256']!=sha(source/'request.json'):raise ValueError('Source receipt ancestry mismatch')
        verify_hashes(directory,receipt['hashes'])
        records.append((row,source,directory,receipt))
    output.mkdir(parents=True);(output/'frames').mkdir()
    request=deepcopy(a);request['inventory']=[deepcopy(new) if r['frame_id']==frame else r for r in a['inventory']]
    request['script_hashes'][Path(__file__).name]=sha(__file__)
    request['composition']={'parent':str(parent),'parent_request_sha256':sha(parent/'request.json'),
                            'replacement':str(replacement),'replacement_request_sha256':sha(replacement/'request.json'),
                            'replacement_frame':frame,'unchanged_reused_frames':149,'rerender_claimed':False}
    atomic_json(output/'request.json',request)
    for row,source,directory,receipt in records:
        target=output/'frames'/row['frame_id'];target.mkdir()
        for name in receipt['hashes']:
            p,q=directory/name,target/name;q.parent.mkdir(parents=True,exist_ok=True)
            try:os.link(p,q)
            except OSError:shutil.copyfile(p,q)
        verify_hashes(target,receipt['hashes'])
        inherited={'kind':'hash_verified_reuse','source_request':str(source/'request.json'),
                   'source_request_sha256':sha(source/'request.json'),'source_receipt':str(directory/'complete.json'),
                   'source_receipt_sha256':sha(directory/'complete.json'),'not_a_new_render':True}
        atomic_json(target/'complete.json',{'request_sha256':sha(output/'request.json'),'hashes':receipt['hashes'],'ancestry':inherited})
    for group in (parent/'contact_sheets').iterdir():
        if frame not in read(group/'manifest.json')['render_hashes']:link_tree(group,output/'contact_sheets'/group.name)
    reviewdir=output/'visual_reviews';reviewdir.mkdir()
    for path in (parent/'visual_reviews').glob('*.json'):
        review=read(path)
        if frame in review['frame_ids']:continue
        review['reused_review']={'source':str(path),'sha256':sha(path),'pixels_unchanged':True}
        atomic_json(reviewdir/path.name,review)
    atomic_json(output/'progress.json',{'stage':'composed_verified_frames_requires_updated_visual_review','complete':150,
                                      'reused':149,'replacement':frame})
    audit_ancestry(output)


def audit_ancestry(output):
    request=verify_request(output);ancestry=request['composition'];parent=read(Path(ancestry['parent'])/'request.json')
    replacement=read(Path(ancestry['replacement'])/'request.json');assert_texture_equivalent(parent,replacement)
    ids=request['ordered_frame_ids']
    if len(ids)!=150 or len(set(ids))!=150 or ids!=parent['ordered_frame_ids'] or [r['frame_id'] for r in request['inventory']]!=ids:
        raise ValueError('Derived temporal inventory mismatch')
    if request['recipe']!=parent['recipe'] or request['source_rows']!=parent['source_rows']:
        raise ValueError('Derived recipe or source inputs changed')
    for name in ['parent','replacement']:
        if sha(Path(ancestry[name])/'request.json')!=ancestry[name+'_request_sha256']:raise ValueError('Ancestor request changed')
    for row in request['inventory']:
        directory=output/'frames'/row['frame_id'];receipt=read(directory/'complete.json');origin=receipt['ancestry']
        expected_root=Path(ancestry['replacement'] if row['frame_id']==ancestry['replacement_frame'] else ancestry['parent'])
        if origin['source_receipt']!=str(expected_root/'frames'/row['frame_id']/'complete.json'):raise ValueError('Wrong source time in receipt ancestry')
        if origin['source_request']!=str(expected_root/'request.json'):raise ValueError('Wrong request in receipt ancestry')
        if sha(origin['source_receipt'])!=origin['source_receipt_sha256'] or sha(origin['source_request'])!=origin['source_request_sha256']:
            raise ValueError('Ancestor hash mismatch')
        source_request=read(origin['source_request']);source_record=next(r for r in source_request['inventory'] if r['frame_id']==row['frame_id'])
        if source_record!=row or read(origin['source_receipt'])['hashes']!=receipt['hashes']:raise ValueError('Reused inputs/outputs differ')
        if receipt['request_sha256']!=sha(output/'request.json'):raise ValueError('Derived request mismatch')
        verify_hashes(directory,receipt['hashes'])
    return {'status':'pass','verified_times':150,'unchanged_reused':149,'replacement':ancestry['replacement_frame']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['compose','audit'])
    p.add_argument('--parent',type=Path);p.add_argument('--replacement',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.action=='compose':
        if a.parent is None or a.replacement is None:p.error('--parent and --replacement required')
        compose(a.parent,a.replacement,a.output)
    else:print(audit_ancestry(a.output))
