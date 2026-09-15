"""Freeze the two-time geometric progress without claiming full-video completion."""
import argparse
import hashlib
import inspect
from pathlib import Path
import shutil
import subprocess
import numpy as np
from joint_temporal_texture import read,sha,atomic_json
from study_close_boundary_completion import ROOT,transform
from review_jaw_repair_transfer import verified_image
from render_smooth_temporal_mesh_video import verify_request
import study_poisson_jaw_completion as raw


def run(check=False):
    manifest=ROOT/'artifact_manifest.json'
    if check:
        files=read(manifest)['files']
        for path,digest in files.items():
            if sha(path)!=digest:raise ValueError('Changed artifact: '+path)
        print('Verified',len(files),'hashes');return
    if manifest.exists():raise ValueError('Already frozen')
    inspected=[];external={};frames=[];renders=0
    executed=hashlib.sha256(transform(inspect.getsource(raw.prepare)).encode()).hexdigest()
    for frame in ['001193','001195']:
        root=ROOT/frame;controller=read(root/'controller_request.json')
        if controller['proposal_execution_sha256']!=executed:raise ValueError('Changed proposal adapter')
        if controller['source_sha256']!=sha(Path(__file__).with_name('study_close_boundary_completion.py')):raise ValueError('Changed controller')
        candidate=root/'interpolated'/frame;result=read(candidate/'result.json');audit=read(candidate/'audit.json')
        if not result['observed_guard_passed'] or audit['mesh_sha256']!=sha(candidate/'mesh.ply') or len(audit['native_ray_checks'])!=124:
            raise ValueError('Independent geometry audit missing')
        if not audit['original_prefix_exact'] or not audit['local_certificates_recomputed']:raise ValueError('Weak geometry evidence')
        matched=read(root/'matched_gap/result.json')
        if not matched['observed_guard_passed'] or matched['rounds'][-1]['removed'] or len(matched['rounds'][-1]['checks'])!=124:
            raise ValueError('Matched safety audit missing')
        for view in ['moving','F004_E005_1210FP']:
            for variant in ['baseline','previous','close_boundary','matched_gap']:
                if view=='moving' and variant=='baseline':continue
                location=root/'rgb'/view/variant;q=verify_request(location)
                _,r=verified_image(location,frame);renders+=1
                for row in q['inventory']:
                    for key in ['mesh','metadata']:
                        if sha(row[key])!=row[key+'_sha256']:raise ValueError('Changed geometry input')
                        external[row[key]]=row[key+'_sha256']
                for name,digest in q['script_hashes'].items():external[str(Path(__file__).with_name(name))]=digest
            inspected += [root/'review'/(view+'_'+detail+'.png') for detail in ['head','detail']]
            inspected.append(root/'matched_review'/(view+'_detail.png'))
        frames.append(dict(frame=frame,mesh=str(candidate/'mesh.ply'),sha256=sha(candidate/'mesh.ply'),added=result['added']))
    _,held=verified_image(ROOT/'heldout/close_boundary','001193');renders+=1
    if renders!=15:raise ValueError('Wrong experiment inventory')
    metrics=read(ROOT/'heldout/metrics.json')
    if metrics['full_frame_metrics'] or metrics['loss_reported']:raise ValueError('Wrong metrics')
    for key in ['face_psnr','face_ssim','face_lpips']:
        if not np.isfinite(metrics['rows'][0][key]) or metrics['rows'][0][key]!=metrics['rows'][1][key]:raise ValueError('Expected verified face non-regression')
    inspected.append(ROOT/'heldout/comparison.png')
    atomic_json(ROOT/'visual_review.json',dict(status='local_jaw_mesh_improvement_residual_artifacts',
        reviewer='main_agent_actual_image_inspection',inspected={str(p):sha(p) for p in inspected},
        supersedes_provisional_pending_labels=True,artifact_free=False,video_changed=False,
        notes='001193 under-jaw hole visibly reduced relative to production; no broad head deterioration seen. Same-raw control confirms tiny incremental closure. 001195 gains are smaller. One movie RGB pixel and native edge holes remain; crown/hair and hand/lipstick are not repaired.'))
    processes=subprocess.check_output(['ps','-eo','pid,args'],text=True)
    active=[line for line in processes.splitlines() if 'python' in line and '/bin/bash' not in line and any(n in line for n in [
        'study_close_boundary_completion.py prepare','render_close_boundary_completion.py render',
        'review_matched_gap_completion.py render','score_close_boundary_heldout.py'])]
    if active:raise ValueError('Worker still active: '+str(active))
    atomic_json(ROOT/'supervision_final.json',dict(workers_alive=False,new_renders=15,frames=frames,
        free_bytes=shutil.disk_usage(ROOT).free,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
        initial_scorer_failure='Texture-only scorer rejects changed meshes. Geometry-aware scorer completed; failed log retained.'))
    scripts=Path(__file__).parent;repo=scripts.parent;tests=Path('/mnt/data/dec5_close_boundary_tests.log')
    if '6 passed' not in tests.read_text():raise ValueError('Focused tests missing')
    sources=[scripts/n for n in ['diagnose_residual_jaw_proposals.py','study_close_boundary_completion.py',
        'render_close_boundary_completion.py','build_matched_gap_control.py','review_matched_gap_completion.py',
        'score_close_boundary_heldout.py','freeze_close_boundary_completion.py']]
    sources += [repo/'tests/test_close_boundary_completion.py',repo/'experiments/dec5_close_boundary_completion.md',tests]
    logs=list(Path('/mnt/data').glob('dec5_close_boundary_*.log'))+list(Path('/mnt/data').glob('dec5_matched_gap_*.log'))
    logs += [Path('/mnt/data/dec5_residual_jaw_verified.log')]
    files={str(p):sha(p) for base in [ROOT,Path('/mnt/data/dec5_residual_jaw_proposals_verified')]
        for p in base.rglob('*') if p.is_file() and p!=manifest}
    for path,digest in external.items():
        if sha(path)!=digest:raise ValueError('Changed dependency: '+path)
    files.update(external);files.update({str(p):sha(p) for p in [*sources,*logs]})
    atomic_json(manifest,dict(files=files,status='verified_local_geometry_progress_not_full_goal',
        new_rgb_controls=renders,heldout_face_unchanged=True,production_video_changed=False,artifact_free=False))
    print('Frozen',len(files),'hashes',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');run(p.parse_args().check)
