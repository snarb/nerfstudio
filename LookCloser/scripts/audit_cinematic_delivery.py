"""Read-only publication checks, plus a compact delivery receipt for each film.

The authoritative visual verdict is the terminal publication/manual review,
not the compositor's immutable pre-review `visual_status` placeholder.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile
from joint_temporal_texture import read,sha,atomic_json

VARIANTS=['locked_arc','free_arc','rising_arc','soft_diagonal']


def audit(base,variants):
    expected=[f'{899+2*i:06d}' for i in range(150)];records=[];bindings={}
    for variant in variants:
        root=base/variant;publication=read(root/'publication.json');out=root/'presentation'
        assert publication['status']=='reviewed_hybrid_cinematic_choice_with_known_residuals'
        assert not publication['all_frames_are_3d_renders'] and not publication['artifact_free_approval']
        for name,digest in publication['bindings'].items():
            assert sha(root/name)==digest;bindings[str(root/name)]=digest
        review=read(root/'manual_visual_review.json')
        assert review['status']=='reviewed_hybrid_choice_with_known_residuals'
        assert review['overview_groups_inspected']==list(range(0,150,10))
        for name,digest in review['inspected_image_hashes'].items():
            assert sha(root/name)==digest;bindings[str(root/name)]=digest
        request=read(root/'request.json');complete=read(out/'complete.json');video=read(out/'video_manifest.json')
        assert request['ordered_frame_ids']==complete['ordered_frame_ids']==expected
        assert [row['frame_id'] for row in complete['records']]==expected
        assert [row['index'] for row in complete['records']]==list(range(150))
        assert complete['frame_count']==150 and complete['pure_train_count']==24 and complete['dissolve_count']==8
        assert not complete['all_frames_are_3d_renders'] and not video['all_frames_are_3d_renders']
        assert {p.parent.name for p in (root/'frames').glob('*/complete.json')}==set(expected[:126])
        assert {p.parent.name for p in (out/'frames').glob('*/complete.json')}==set(expected)
        for frame,row in zip(expected,complete['records']):
            receipt=read(out/'frames'/frame/'complete.json');assert receipt==row
            path=out/'frames'/frame/'frame.png';assert sha(path)==row['image_sha256']
            bindings[str(path)]=row['image_sha256']
        with zipfile.ZipFile(out/'frames.zip') as archive:
            assert archive.testzip() is None
            names=archive.namelist()
            assert len(names)==len(set(names))==150
            for frame,row in zip(expected,complete['records']):
                name=f'{frame}.png'
                # Existing encoder stores source IDs as flat file basenames.
                assert name in names, (name,names[:3])
                assert hashlib.sha256(archive.read(name)).hexdigest()==row['image_sha256']
        env=dict(os.environ,LD_PRELOAD='/lib/x86_64-linux-gnu/libmpg123.so.0')
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=width,height,nb_read_frames,r_frame_rate,duration','-of','json',str(out/'video.mp4')],env=env,text=True))['streams'][0]
        assert probe['width']==1080 and probe['height']==1920 and probe['r_frame_rate']=='24/1'
        assert int(probe['nb_read_frames'])==150 and abs(float(probe['duration'])-6.25)<1e-6
        assert sha(out/'video.mp4')==video['video_sha256'] and sha(out/'frames.zip')==video['frames_zip_sha256']
        records.append(dict(variant=variant,video=str(out/'video.mp4'),frames_zip=str(out/'frames.zip'),
            video_sha256=video['video_sha256'],frames_zip_sha256=video['frames_zip_sha256'],
            publication_sha256=sha(root/'publication.json'),frame_count=150,fps=24,duration_seconds=6.25,
            pure_actual_train_ending_frames=24,known_residuals=review['known_residuals'],artifact_free=False))
    for name in ['independent_path_audit.json','independent_real_ending_audit.json']:
        path=base/name;bindings[str(path)]=sha(path)
    if variants==VARIANTS:
        destination=base/'delivery_audit.json'
    else:
        assert len(variants)==1;destination=base/variants[0]/'delivery_audit.json'
    atomic_json(destination,dict(status='delivery_integrity_pass_with_disclosed_visual_residuals',
        records=records,hashes=bindings,script_sha256=sha(__file__),all_four=variants==VARIANTS,
        real_source_ending_not_mesh_repair=True,quality_metrics=False))
    print('Independently verified delivery',variants,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--variant',choices=VARIANTS);a=p.parse_args()
    audit(a.base,[a.variant] if a.variant else VARIANTS)
