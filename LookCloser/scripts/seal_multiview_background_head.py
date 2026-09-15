"""Seal an actually reviewed pruning diagnostic without production promotion."""
import argparse
from pathlib import Path
import subprocess
import numpy as np
from joint_temporal_texture import read,sha,atomic_json
from prune_multiview_background_head import ROOT,INSET,MOVIE,MASKS,FRAMES
from review_jaw_repair_transfer import verified_image


def run():
    visual=read(ROOT/'visual_review.json');assert visual['operator']=='main_LLM_actual_image_inspection'
    assert not visual['production_promoted'] and not visual['artifact_free_approval']
    for p,h in visual['inspected_images'].items():assert sha(ROOT/p)==h
    live=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
    assert not [x for x in live if 'python scripts/study_multiview_background_head.py render ' in x and '/bin/bash' not in x]
    hashes={};review=read(ROOT/'review/result.json');assert len(review['records'])==8
    for frame in FRAMES:
        q=read(ROOT/frame/'request.json');a=read(ROOT/frame/'audit.json');r=read(ROOT/frame/'result.json')
        assert a['request_sha256']==r['request_sha256']==sha(ROOT/frame/'request.json')
        assert a['result_sha256']==sha(ROOT/frame/'result.json')
        for p,h in q['scripts'].items():assert sha(p)==h;hashes[p]=h
        mr=read(MASKS/frame/'request.json')
        for p,h in mr['rgb_receipt']['source_rgb_hashes'].items():assert sha(p)==h;hashes[p]=h
        depth=q['depth_receipt'];cal=read(depth['transforms']);mapping={r['physical_camera']:r for r in cal['frames']}
        for name,h in depth['depth_sha256'].items():
            p=Path(depth['dense'])/'stereo/depth_maps'/(mapping[name]['file_path']+'.geometric.bin')
            assert sha(p)==h;hashes[str(p)]=h
        for view in ['moving','native_unmasked']:
            roots=[MOVIE if view=='moving' else INSET/'rgb'/frame/view/'baseline',
                INSET/'rgb'/frame/view/'completion',ROOT/'rgb'/frame/view/'pruned',ROOT/'rgb'/frame/view/'pruned_inset']
            qs=[];rs=[]
            for root in roots:
                image,result=verified_image(root,frame);qs.append(read(root/'request.json'));rs.append(result)
                assert image.shape==(1920,1080,3) and np.isfinite(image).all()
                assert len(set(result['source_cameras']))==62
                for path in [root/'request.json',root/'frames'/frame/'complete.json']:
                    hashes[str(path)]=sha(path)
                for name,h in read(root/'frames'/frame/'complete.json')['hashes'].items():hashes[str(root/'frames'/frame/name)]=h
            entries=[next(e for e in q['inventory'] if e['frame_id']==frame) for q in qs]
            for i in range(1,4):
                for key in ['camera','source_cameras','fixed_exposure']:assert rs[0][key]==rs[i][key]
                for key in ['profiles_sha256','exposure_sha256','calibration_sha256']:assert qs[0][key]==qs[i][key]
                assert entries[0]['source_masks']==entries[i]['source_masks']
            for i,arm in [(2,'pruned'),(3,'pruned_inset')]:
                assert rs[i]['mesh_sha256']==sha(ROOT/frame/arm/'mesh.ply')==r['hashes'][arm+'/mesh.ply']
    atomic_json(ROOT/'completion.json',dict(workers_terminal=True,matched_render_count=8,
        visual_review_sha256=sha(ROOT/'visual_review.json'),review_result_sha256=sha(ROOT/'review/result.json'),
        production_promoted=False,quality_metrics_computed=False,combined_newly_exposed_shell_guard_passed=False))
    for p in ROOT.rglob('*'):
        if p.is_file() and p.name!='artifact_manifest.json':hashes[str(p)]=sha(p)
    for name in ['prune_multiview_background_head.py','study_multiview_background_head.py',Path(__file__).name]:
        p=Path(__file__).resolve().with_name(name);hashes[str(p)]=sha(p)
    atomic_json(ROOT/'artifact_manifest.json',dict(hashes=hashes,production_promoted=False,artifact_free_approval=False))
    print('Sealed',len(hashes),'bindings',flush=True)


def check():
    hashes=read(ROOT/'artifact_manifest.json')['hashes']
    for p,h in hashes.items():assert sha(p)==h,p
    print('Rechecked',len(hashes),'bindings',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    check() if a.check else run()
