"""Package the four audited movies, explicitly labeling the actual-video ending."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile
from joint_temporal_texture import read,sha,atomic_json

NAMES={'locked_arc':'01_locked_arc.mp4','free_arc':'02_free_arc_beauty.mp4',
       'rising_arc':'03_rising_arc.mp4','soft_diagonal':'04_wide_arc_beauty.mp4'}
README='''DEC5 cinematic choices — 150 dynamic source times, 24 fps, 6.25 seconds each.

01 locked_arc: camera points toward the subject; curved physical approach,
whole-head portrait ending.
02 free_arc_beauty: same approach with an off-center composition gesture and
tighter beauty framing. Suggested first choice.
03 rising_arc: a different gentle diagonal arc, whole-head portrait ending.
04 wide_arc_beauty: a wider angular arc, less physical radial approach,
tighter beauty framing. Suggested alternative.

IMPORTANT: The final second is actual moving footage from train camera
H004_C005_1210SZ, not a 3D reconstruction. The camera and virtual lens come
to rest before an eight-frame dissolve; the original source background
appears during that dissolve. All 24 final frames come from the matching
real source times. There is no frozen actor, generated image or extra
slow-motion version. Zoom and sensor framing are distinct from physical
camera translation.

The preceding 3D part still contains known reconstruction residuals:
rough hair contours, occasional small lipstick/hand membranes and thin
facial seams. The live-action ending bypasses these; it does not repair
the mesh. Some prior silhouette is visible briefly during the dissolve.
Whole-head endings show a studio stand; beauty endings leave it outside
the field of view and intentionally cut off the top of the hair.

The adjacent manifest records original paths and SHA-256 hashes. Separate
full-resolution frame archives remain under each variant's presentation/.
'''


def bundle(base):
    audit=read(base/'delivery_audit.json')
    assert audit['all_four'] and audit['status']=='delivery_integrity_pass_with_disclosed_visual_residuals'
    assert [r['variant'] for r in audit['records']]==list(NAMES)
    records=[]
    for row in audit['records']:
        assert sha(row['video'])==row['video_sha256']
        assert sha(base/row['variant']/'publication.json')==row['publication_sha256']
        records.append(dict(row,archive_name=NAMES[row['variant']]))
    manifest=dict(records=records,delivery_audit_sha256=sha(base/'delivery_audit.json'),
        script_sha256=sha(__file__),last_second_is_real_train_video=True,
        all_frames_are_3d_renders=False,artifact_free_approval=False)
    target=base/'cinematic_choices.zip';partial=base/'cinematic_choices.partial.zip'
    with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_STORED) as z:
        for row in records:z.write(row['video'],row['archive_name'])
        z.writestr('README.txt',README)
        z.writestr('manifest.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    with zipfile.ZipFile(partial) as z:
        assert z.testzip() is None and len(z.namelist())==6
        for row in records:assert hashlib.sha256(z.read(row['archive_name'])).hexdigest()==row['video_sha256']
    os.replace(partial,target)
    atomic_json(base/'cinematic_choices_bundle.json',dict(manifest,archive=str(target),
        archive_sha256=sha(target),archive_bytes=target.stat().st_size))
    print(target,target.stat().st_size,'bytes; all4 archived MP4 hashes rechecked',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    bundle(p.parse_args().base)
