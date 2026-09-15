"""Verify and package the two wide-spiral presentation movies after review."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile
from joint_temporal_texture import read,sha,atomic_json

NAMES={'wide_spiral_lookat':'01_wide_spiral_lookat.mp4',
       'wide_spiral_free':'02_wide_spiral_free.mp4'}
README='''Two wide first-turn spiral choices. 150 moving source times, 24 fps,
6.25 seconds, portrait1080x1920. 01 continuously aims toward the woman;
02 adds controlled composition drift (recommended to see the movement).

This is a contracting ellipse across the interior FRONT camera rig, not a
360-degree orbit behind the woman. Camera translation is distinct from zoom.
One broad turn is completed before convergence to physical train camera H/C.

The last second is24 actual moving train RGB images, following an8-frame
display dissolve at a fixed camera/lens. This is an explicitly requested
presentation edit, NOT mesh repair. Original room background appears there.
Earlier frames use unchanged production meshes and hard train RGB sampling.
Known residuals remain in hair contours and small facial/hand/lipstick seams.
Read report.md for measured camera motion, controls and visual limitations.
'''


def bundle(base):
    expected=[f'{899+2*i:06d}' for i in range(150)];records=[]
    previous=Path('/mnt/data/dec5_cinematic_pushin_v4')
    independent=read(previous/'independent_real_ending_audit.json')
    assert independent['status']=='all128_real_rgb_frames_verified'
    for variant,name in NAMES.items():
        root=base/variant;out=root/'presentation';pub=read(root/'publication.json')
        assert pub['status']=='reviewed_wide_spiral_hybrid_with_known_residuals'
        for file,digest in pub['bindings'].items():assert sha(root/file)==digest
        review=read(root/'manual_visual_review.json')
        assert review['status']=='reviewed_hybrid_choice_with_known_residuals'
        assert review['overview_groups_inspected']==list(range(0,150,10))
        assert 'presentation/decoded_overview.png' in review['inspected_image_hashes']
        for file,digest in review['inspected_image_hashes'].items():assert sha(root/file)==digest
        request=read(root/'request.json');done=read(out/'complete.json');video=read(out/'video_manifest.json')
        assert request['ordered_frame_ids']==done['ordered_frame_ids']==expected
        assert done['pure_train_count']==24 and done['dissolve_count']==8
        # Same optics/physical H/C ending as independently float64-replayed v4.
        # Compare all32 prepared images, not only the final still or pose labels.
        for frame in expected[118:]:
            old=previous/'free_arc'/'train_ending'/'frames'/frame/'frame.png'
            new=root/'train_ending'/'frames'/frame/'frame.png'
            assert sha(old)==independent['hashes'][str(old)]==sha(new)
        assert {p.parent.name for p in (root/'frames').glob('*/complete.json')}==set(expected[:126])
        assert {p.parent.name for p in (out/'frames').glob('*/complete.json')}==set(expected)
        with zipfile.ZipFile(out/'frames.zip') as z:
            assert z.testzip() is None and len(z.namelist())==150
            for i,(frame,row) in enumerate(zip(expected,done['records'])):
                assert row['frame_id']==frame and row['index']==i
                assert read(out/'frames'/frame/'complete.json')==row
                assert sha(out/'frames'/frame/'frame.png')==row['image_sha256']
                assert hashlib.sha256(z.read(frame+'.png')).hexdigest()==row['image_sha256']
        env=dict(os.environ,LD_PRELOAD='/lib/x86_64-linux-gnu/libmpg123.so.0')
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=width,height,nb_read_frames,r_frame_rate,duration','-of','json',str(out/'video.mp4')],env=env,text=True))['streams'][0]
        assert probe['width']==1080 and probe['height']==1920 and probe['r_frame_rate']=='24/1'
        assert int(probe['nb_read_frames'])==150 and abs(float(probe['duration'])-6.25)<1e-6
        assert sha(out/'video.mp4')==video['video_sha256'] and sha(out/'frames.zip')==video['frames_zip_sha256']
        records.append(dict(variant=variant,archive_name=name,video=str(out/'video.mp4'),video_sha256=video['video_sha256'],
            publication_sha256=sha(root/'publication.json'),frames_zip=str(out/'frames.zip'),frames_zip_sha256=video['frames_zip_sha256'],
            motion_audit_sha256=sha(root/'motion_audit.json'),review_sha256=sha(root/'manual_visual_review.json')))
    manifest=dict(status='two_reviewed_spiral_presentations_integrity_pass',records=records,
        script_sha256=sha(__file__),frame_count_each=150,fps=24,duration_seconds=6.25,
        all64_prepared_endings_equal_independently_replayed_v4=True,
        independent_ending_audit_path=str(previous/'independent_real_ending_audit.json'),
        independent_ending_audit_sha256=sha(previous/'independent_real_ending_audit.json'),
        all_frames_are_3d_renders=False,last_second_is_actual_train_RGB=True,artifact_free_approval=False)
    atomic_json(base/'delivery_audit.json',manifest)
    partial=base/'wide_spiral_choices.partial.zip';target=base/'wide_spiral_choices.zip'
    with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_STORED) as z:
        for row in records:z.write(row['video'],row['archive_name'])
        z.writestr('README.txt',README);z.writestr('manifest.json',json.dumps(manifest,indent=2)+'\n')
        z.write(base/'wide_spiral_free'/'report.md','report.md')
        z.write(base/'wide_spiral_free'/'motion_plot.png','motion_plot.png')
    with zipfile.ZipFile(partial) as z:
        assert z.testzip() is None and len(z.namelist())==6
        for row in records:assert hashlib.sha256(z.read(row['archive_name'])).hexdigest()==row['video_sha256']
    os.replace(partial,target)
    atomic_json(base/'bundle.json',dict(manifest,archive=str(target),archive_sha256=sha(target),archive_bytes=target.stat().st_size))
    print(target,target.stat().st_size,'bytes; two MP4s and300 archived PNG hashes checked',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    bundle(p.parse_args().base)
