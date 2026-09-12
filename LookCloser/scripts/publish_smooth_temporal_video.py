"""Publish portable PNG/video/review artifacts only after the full temporal audit."""
from pathlib import Path
import argparse
import os
import shutil
import tempfile
from joint_temporal_texture import read,sha,atomic_json
from finalize_local_mesh_repair import verify_hashes
from audit_smooth_temporal_video import audit


def publish(source,destination):
    audit(source);report=read(source/'audit.json')
    if report['catastrophic_frame_count']:raise ValueError('Do not publish a failed viewing candidate as selected')
    if destination.exists():
        receipt=read(destination/'publication.json')
        if receipt['source_audit_sha256']!=sha(source/'audit.json'):raise ValueError('Immutable publication mismatch')
        verify_hashes(destination,receipt['hashes']);return
    stage=Path(tempfile.mkdtemp(prefix=destination.name+'.staging-',dir=destination.parent))
    if stage.stat().st_mode & 0o005 != 0o005:stage.chmod(0o755)
    def retain(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(src,dst)
        except OSError:shutil.copyfile(src,dst)
        if sha(src)!=sha(dst):raise ValueError('Publication checksum mismatch')
    retain(source/'smooth_temporal_150.mp4',stage/'video.mp4')
    for folder in ['video_frames','contact_sheets','visual_reviews']:
        for path in sorted((source/folder).rglob('*')):
            if path.is_file():retain(path,stage/path.relative_to(source))
    for name in ['audit.json','frames_audit.csv','request.json','video_manifest.json',
                 'encoded_review.json','encoded_temporal_overview.png','temporal_visual_review.json']:
        retain(source/name,stage/name)
    request=read(source/'request.json')
    for record in request['inventory']:
        frame=source/'frames'/record['frame_id']
        for name in ['result.json','complete.json']:
            retain(frame/name,stage/'render_receipts'/record['frame_id']/name)
    experiment=Path(__file__).resolve().parents[1]/'experiments/dec5_smooth_temporal_mesh_video.md'
    shutil.copyfile(experiment,stage/'report.md')
    atomic_json(stage/'publication.json',{'source_workspace':str(source),'source_audit_sha256':sha(source/'audit.json'),
        'status':'accepted_with_known_artifacts','strict_artifact_free':False,'source_frame_count':150,
        'geometry_included':False,'geometry_note':'Existing meshes remain at the audited paths in request.json; this is the video/PNG download bundle',
        'render_receipts_note':'Historical render receipts also name diagnostic files retained only in the source workspace',
        'hashes':{str(p.relative_to(stage)):sha(p) for p in sorted(stage.rglob('*')) if p.is_file()}})
    verify_hashes(stage,read(stage/'publication.json')['hashes']);os.rename(stage,destination)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);args=p.parse_args();publish(args.source,args.destination)
