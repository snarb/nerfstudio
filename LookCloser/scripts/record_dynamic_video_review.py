"""Record an explicit human/LLM verdict AFTER inspecting a named native sheet group."""
from __future__ import annotations
import argparse
from pathlib import Path
from joint_temporal_texture import read,sha,atomic_json


def record(output,group,notes,status):
    if not notes.strip() or status not in {'pass','accepted_known_artifacts','fail','uncertain'}:
        raise ValueError('Explicit notes and visual status required')
    root=output/'contact_sheets'/group;manifest=read(root/'manifest.json')
    evidence=[dict(path=str(root/f'{name}.png'),sha256=sha(root/f'{name}.png')) for name in ['face','ear_hair','lipstick_hand']]
    for frame,digest in manifest['render_hashes'].items():
        if sha(output/'frames'/frame/'frame.png')!=digest:raise ValueError('Stale review sheet')
        target=output/'visual_reviews'/f'{frame}.json'
        previous=read(target) if target.exists() else None
        value=dict(frame_id=frame,render_sha256=digest,reviewer='LLM_native_image_inspection',
            status=status,artifact_free=status=='pass',notes=notes,evidence=evidence)
        if previous is not None and previous!=value:value['previous_review']=previous
        atomic_json(target,value)
    atomic_json(root/'visual_review.json',dict(status=status,notes=notes,evidence=evidence,frames=list(manifest['render_hashes'])))
    print(f'reviewed={group} frames={len(manifest["render_hashes"])} status={status}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--group',required=True);p.add_argument('--notes',required=True);p.add_argument('--status',required=True)
    a=p.parse_args();record(a.output,a.group,a.notes,a.status)
