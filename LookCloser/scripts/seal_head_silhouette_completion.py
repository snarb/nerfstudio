"""Observe and seal the actually inspected head silhouette experiment."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
from joint_temporal_texture import read,sha,atomic_json
from extend_head_silhouette_guard import ROOT,SOURCE
from probe_head_silhouette_completion import FRAMES,INSET,MASKS,MOVIE
from review_jaw_repair_transfer import verified_image
from render_head_silhouette_cinematic import ROOT as CINE,BASE as CINE_BASE,VARIANTS

SCRIPTS=['probe_head_silhouette_completion.py','study_head_silhouette_completion.py','extend_head_silhouette_guard.py',
         'render_head_silhouette_cinematic.py']


def processes():
    lines=subprocess.check_output(['ps','-eo','pid,etimes,rss,args'],text=True).splitlines()
    return [line.strip() for line in lines if any('/python scripts/'+n in line for n in SCRIPTS) and '/bin/bash' not in line]


def monitor():
    status=dict(utc=datetime.now(timezone.utc).isoformat(),processes=processes(),free_bytes=shutil.disk_usage(ROOT).free,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True).strip(),
        gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip(),
        stages={},logs={},cinematic_complete={v:{f:(CINE/v/'frames'/f/'complete.json').exists() for f in FRAMES} for v in VARIANTS})
    for frame in FRAMES:
        p=ROOT/frame/'guarded'/'result.json'
        status['stages'][frame]=dict(depth_guard_passed=read(p)['native_free_space_guard_passed'],
            rgb_complete={v:(ROOT/'rgb'/frame/v/'frames'/frame/'complete.json').exists() for v in ['moving','native_unmasked']})
        p=SOURCE/('render_'+frame+'.log')
        if p.exists():status['logs'][frame]=p.read_text().splitlines()[-5:]
    with (ROOT/'checks.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
    print(json.dumps(status),flush=True)


def seal():
    assert not processes(),'Worker still live'
    review=read(ROOT/'visual_review.json');assert review['operator']=='main_LLM_actual_image_inspection'
    assert not review['production_promoted'] and not review['artifact_free_approval']
    for p,h in review['inspected_images'].items():assert sha(p)==h,p
    result=read(ROOT/'review'/'result.json');assert len(result['records'])==4 and not result['quality_metrics']
    hashes={}
    def bind(p,h=None):
        actual=sha(p)
        if h is not None:assert actual==h,str(p)
        hashes[str(p)]=actual
    for frame in FRAMES:
        q=read(ROOT/frame/'request.json');r=read(ROOT/frame/'result.json');guard=ROOT/frame/'guarded'
        assert r['request_sha256']==sha(ROOT/frame/'request.json')
        for mapping in [q['dependencies'],q['scripts']]:
            for p,h in mapping.items():bind(p,h)
        bind(q['source_mesh'],q['source_mesh_sha256'])
        for p,h in r['hashes'].items():bind(ROOT/frame/p,h)
        g=read(guard/'result.json');gq=read(guard/'request.json');a=read(guard/'audit.json')
        assert g['native_free_space_guard_passed'] and a['result_sha256']==sha(guard/'result.json')
        assert len(a['checks'])==124 and all(x['trusted_free_pixels']==0 for x in a['checks'])
        bind(guard/'request.json',g['request_sha256'])
        bind(gq['prior_guard_result'],gq['prior_guard_result_sha256'])
        for mapping in [gq['scripts'],gq['frozen_proposal_links']]:
            for p,h in mapping.items():bind(p,h)
        for p,h in g['hashes'].items():bind(guard/p,h)
        maskreq=read(MASKS/frame/'request.json')
        for p,h in maskreq['rgb_receipt']['source_rgb_hashes'].items():bind(p,h)
        receipt=gq['depth_receipt'];cal=read(receipt['transforms']);mapping={r['physical_camera']:r for r in cal['frames']}
        bind(receipt['transforms'])
        for camera,h in receipt['depth_sha256'].items():
            bind(Path(receipt['dense'])/'stereo/depth_maps'/(mapping[camera]['file_path']+'.geometric.bin'),h)
        for view in ['moving','native_unmasked']:
            baseline=MOVIE if view=='moving' else INSET/'rgb'/frame/view/'baseline'
            roots=[baseline,INSET/'rgb'/frame/view/'completion',ROOT/'rgb'/frame/view]
            qs=[];rs=[]
            for root in roots:
                image,record=verified_image(root,frame);rs.append(record);qs.append(read(root/'request.json'))
                assert image.shape==(1920,1080,3) and np.isfinite(image).all()
                assert len(set(record['source_cameras']))==62
                bind(root/'request.json');bind(root/'frames'/frame/'complete.json')
                for p,h in read(root/'frames'/frame/'complete.json')['hashes'].items():bind(root/'frames'/frame/p,h)
            for i in [1,2]:
                for k in ['camera','source_cameras','fixed_exposure']:assert rs[0][k]==rs[i][k]
                for k in ['profiles_sha256','exposure_sha256','calibration_sha256']:assert qs[0][k]==qs[i][k]
                entries=[next(e for e in q['inventory'] if e['frame_id']==frame) for q in [qs[0],qs[i]]]
                assert entries[0]['source_masks']==entries[1]['source_masks']
            assert rs[2]['mesh_sha256']==sha(guard/'mesh.ply')
            if view=='native_unmasked':assert qs[2]['native_target_mask_disabled']
    cine=read(CINE/'review'/'result.json');assert len(cine['records'])==4 and not cine['quality_metrics']
    for variant in VARIANTS:
        baseline=CINE_BASE/variant;candidate=CINE/variant
        q=read(candidate/'request.json');bind(baseline/'request.json',q['matched_cinematic_parent_sha256'])
        for bindings in q['geometry_bindings'].values():
            for p,h in bindings.items():bind(p,h)
        for frame in FRAMES:
            rs=[]
            for root in [baseline,candidate]:
                image,r=verified_image(root,frame);rs.append(r)
                assert image.shape==(1920,1080,3) and np.isfinite(image).all()
                bind(root/'request.json');bind(root/'frames'/frame/'complete.json')
                for p,h in read(root/'frames'/frame/'complete.json')['hashes'].items():bind(root/'frames'/frame/p,h)
            for k in ['camera','source_cameras','fixed_exposure']:assert rs[0][k]==rs[1][k]
            assert rs[1]['mesh_sha256']==sha(ROOT/frame/'guarded'/'mesh.ply')
    atomic_json(ROOT/'completion.json',dict(workers_terminal=True,rgb_comparisons=8,
        visual_review_sha256=sha(ROOT/'visual_review.json'),review_result_sha256=sha(ROOT/'review'/'result.json'),
        all_native_depth_checks_passed=True,production_promoted=False,quality_metrics_computed=False))
    for root in [SOURCE,ROOT,CINE]:
        for p in root.rglob('*'):
            if p.is_file() and p.name!='artifact_manifest.json':bind(p)
    for name in [*SCRIPTS,Path(__file__).name]:bind(Path(__file__).resolve().with_name(name))
    for relative in ['tests/test_head_silhouette_completion.py','experiments/dec5_head_silhouette_completion.md']:
        bind(Path(__file__).resolve().parents[1]/relative)
    atomic_json(ROOT/'artifact_manifest.json',dict(hashes=hashes,production_promoted=False,artifact_free_approval=False))
    print('Sealed',len(hashes),'bindings',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['monitor','seal','check']);a=p.parse_args()
    if a.action=='check':
        h=read(ROOT/'artifact_manifest.json')['hashes']
        for p,expected in h.items():assert sha(p)==expected,p
        print('Rechecked',len(h),'bindings',flush=True)
    else:globals()[a.action]()
