"""Replay matched RGB integrity and seal the completed two-time visual review."""
from pathlib import Path
from joint_temporal_texture import read,sha,atomic_json
from render_wide_fringe_rgb import ROOT,PARENT,MESH,FRAMES,review
from render_smooth_temporal_mesh_video import verify_request
from review_jaw_repair_transfer import verified_image


def run():
    request=verify_request(ROOT);review()
    result=read(ROOT/'review/result.json');visual=read(ROOT/'visual_review.json')
    assert visual['status']=='partial_fringe_improvement_not_promoted'
    assert sorted(visual['frames'])==FRAMES and len(result['records'])==2
    inputs={str(PARENT/'request.json'):sha(PARENT/'request.json')}
    for frame in FRAMES:
        _,receipt=verified_image(ROOT,frame);verified_image(PARENT,frame)
        assert sha(ROOT/'frames'/frame/'frame.png')==visual['frames'][frame]['candidate_sha256']
        assert next(r for r in result['records'] if r['frame']==frame)['independent_wide_depths_match']
        for name in ('frame.png','target_depth.npz','result.json','complete.json'):
            p=PARENT/'frames'/frame/name;inputs[str(p)]=sha(p)
        for relative in ('request.json','replace/result.json','replace/mesh.ply'):
            p=MESH/frame/relative;inputs[str(p)]=sha(p)
    for row in request['source_rows']:
        source=Path(row['source_dataset'])
        for image in row['source_images']:
            p=source/image['file_path'];assert sha(p)==image['sha256'];inputs[str(p)]=image['sha256']
    for name,digest in request['script_hashes'].items():
        p=Path(__file__).with_name(name);assert sha(p)==digest;inputs[str(p.resolve())]=digest
    inputs[str(Path(__file__).resolve())]=sha(__file__)
    atomic_json(ROOT/'completion.json',dict(frames=FRAMES,request_sha256=sha(ROOT/'request.json'),
        visual_review_sha256=sha(ROOT/'visual_review.json'),rgb_workers_terminal=True,
        partial_result=True,full_video_promoted=False))
    hashes={str(p):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
    hashes.update(inputs);atomic_json(ROOT/'artifact_manifest.json',dict(hashes=hashes))
    for p,digest in read(ROOT/'artifact_manifest.json')['hashes'].items(): assert sha(p)==digest
    print('Sealed and rechecked',len(hashes),'hashes',flush=True)


if __name__=='__main__':run()
