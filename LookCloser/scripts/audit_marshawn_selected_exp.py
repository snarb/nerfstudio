"""Read-only actual completed-exp selection audit, separate from launch permission."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb')as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def write(path,value):
    p=path.with_suffix('.tmp');p.write_text(json.dumps(value,indent=2)+'\n');p.replace(path)


def main(c):
    branch=c/'exp_geometry_candidate_20261106';run=c/'jobs/structural_exp_hplus1/run';raw=(run/'complete.json').read_bytes();complete=json.loads(raw)
    if complete.get('technical_smoke')is not False or complete['final_step']<106316 or not complete['plateau']['stopped']:raise ValueError('Completed production ancestry and plateau required')
    cfg=complete['config']
    if cfg['density_activation']!='trunc_exp'or cfg['correct_sh_directions']is not True or cfg['frequency_projection_mode']!='legacy':raise ValueError('Wrong structural parent')
    step=int(complete['selection']['step']);folder=c/'independent_occupancy'/f'structural_exp_selected_{step:09d}';folder.mkdir(exist_ok=True)
    lock=(folder/'audit.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    if(folder/'selected_audit_complete.json').exists():return
    checkpoint=run/'checkpoints'/f'step-{step:09d}.ckpt';target=folder/'checkpoint.ckpt'
    if not target.exists():os.link(checkpoint,target)
    for source in [run/'worker/lookcloser/frozen/config.yml',run/'occupancy'/f'step-{step:09d}_grid.json',run/'occupancy'/f'step-{step:09d}_grid.npz']:
        if not(folder/source.name).exists():shutil.copyfile(source,folder/source.name)
    shutil.copytree(run/'renders'/f'step-{step:09d}',folder/'renders',dirs_exist_ok=True)
    (folder/'observed_parent_complete.json').write_bytes(raw)
    write(folder/'archive_receipt.json',dict(completed_parent_sha256=hashlib.sha256(raw).hexdigest(),selected_step=step,final_step=complete['final_step'],checkpoint_sha256=sha(target),source_checkpoint=str(checkpoint),preserved_hardlink=True,production_mutated=False,launch_permission_granted=False))
    query=branch/'clothing_103_predeclared.json';audit=branch/'audit_marshawn_tuning_checkpoint.py'
    assert sha(query)=='e11c111e4a41b02417e98ae2e0f4f5f6055883c070539b2334748974bb3c712c'
    assert sha(audit)=='b027a3c9d2cebc2d9ff36be3456d13512602bada5370cabbbd3b356a8786616a'
    frozen=c/'structural_extension_code';sys.path.insert(0,str(frozen/'LookCloser/scripts'))
    from marshawn_tuning_controller import gpu_snapshot,identity,alive
    trace=folder/'clothing103_trace';health_path=folder/'clothing103_producer_health.json'
    if not(trace/'rays/diagnostic_summary.json').exists():
        process=None;before={};producer=None;global_guard=None
        while True:
            devices=gpu_snapshot();gpu=None
            for arm in ('C','B','A'):
                candidate=c/'sh_geometry_candidate_20261106'/('run_'+arm)
                progress=json.loads((candidate/'progress.json').read_text());p=identity(progress['pid'])
                for index,d in devices.items():
                    if p and alive(p)and any(x['pid']==progress['pid']for x in d['processes']):
                        gpu=index;process=p;producer=candidate;before=progress;break
                if gpu is not None:break
            if gpu is None:
                for index,d in devices.items():
                    if d['processes']or d['used_mib']>128:continue
                    held=Path(f'/tmp/lookcloser_tuning_gpu_{os.getuid()}_{index}.lock').open('a+')
                    try:fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:held.close();continue
                    if gpu_snapshot()[index]['processes']:held.close();continue
                    gpu=index;global_guard=held;break
            if gpu is not None:break
            write(folder/'state.json',dict(phase='waiting_idle_gpu',time=time.time()));time.sleep(60)
        guard=Path(f'/tmp/lookcloser_readonly_audit_{os.getuid()}_{gpu}.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX)
        health=dict(before_progress=before,producer_identity=process,before_gpu=devices[gpu],authorized_overlap=process is not None,exclusive_idle_gpu=process is None,memory_cap_gib=10,optimizer_updates=0,no_speed_claim=True)
        write(health_path,health)
        if trace.exists():trace.rename(folder/('incomplete_trace_'+str(int(time.time()))))
        cmd=[sys.executable,str(audit),'trace','--data',str(c/'data'),'--output',str(trace),'--queries',str(query),'--receipt',str(folder/f'step-{step:09d}_grid.json'),
            '--checkpoint',str(target),'--config',str(folder/'config.yml'),'--code-root',str(frozen),'--gpu',str(gpu),'--skip-wholegrid','--allow-readonly-overlap','--memory-cap-gib','10',
            '--expected-sh','true','--expected-density','trunc_exp','--expected-projection','legacy']
        env=dict(os.environ,PYTHONPATH=f'{frozen}:{frozen}/LookCloser/scripts',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        with(folder/'trace.log').open('a')as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=frozen/'LookCloser')
        after=json.loads((producer/'progress.json').read_text())if producer else{};ok=r.returncode==0 and (process is None or(alive(process)and after['step']>=before['step']))
        health.update(after_progress=after,producer_alive_after=alive(process)if process else None,after_gpu=gpu_snapshot()[gpu],audit_environment_healthy=bool(ok),returncode=r.returncode);write(health_path,health)
        guard.close()
        if global_guard:global_guard.close()
        if not ok:raise RuntimeError('Read-only audit/health requires attention, not a training-failure assertion')
    sys.path.insert(0,str(branch))
    from compare_marshawn_exp_geometry_traces import inspect_trace,summarize,GROUPS,PROFILES
    from summarize_marshawn_depth_residuals import main as summarize_depth
    data=json.loads((trace/'rays/audit.json').read_text());chain=json.loads((trace/'coordinate_chain.json').read_text());q=json.loads(query.read_text())
    assert data['step']==step and data['frozen_state_unchanged'] and chain['feature_reweighting_strength']==(1. if step<=75940 else .3)
    assert [r['pixel']for r in data['rows']]==q['pixels']
    rows=[]
    for r in data['rows']:
        p=r['pixel'];profiles={name:inspect_trace(trace/'rays',r,name)for name in PROFILES};v=r['uniform_convergence']
        rows.append(dict(point=p['witness_id'],camera=p['camera_name'],actor=p['actor'],region=p['region'],group=p['diagnostic_group'],profiles=profiles,
            rgb_opacity_converged=v['opacity_abs_difference']<=.005 and v['foreground_rgb_max_abs_difference']<=1/255,
            mass_convergence_delta=float(np.max(abs(np.array(profiles['uniform']['mass'])-profiles['uniform2x']['mass']))),band_mass_convergence_delta=abs(profiles['uniform']['positive_band_unconditional_mass']-profiles['uniform2x']['positive_band_unconditional_mass'])))
    model=dict(source=str(folder),step=step,rows=rows,groups={g:summarize([r for r in rows if r['group']==g])for g in GROUPS},checkpoint_sha256=data['checkpoint_sha256'],coordinate_chain=chain,
        heldout_metrics={k:complete['selection']['metrics'][k]for k in ('psnr','ssim','lpips')})
    result=dict(kind='COMPLETED_PARENT_SELECTED103_CLOTHING_AUDIT',models={'exp_selected':model},files={str(trace/'rays/audit.json'):sha(trace/'rays/audit.json')},summary_script_sha256=sha(Path(__file__)),parent_complete_sha256=hashlib.sha256(raw).hexdigest(),query_sha256=sha(query),
        selection_is_actual_completed_parent=True,group_note='Source groups label future auxiliary roles; this exp parent has no fitted auxiliary geometry labels. All are TRAIN RGB observations.',launch_permission_granted=False,model_promotion_authorized=False)
    write(folder/'comparison.json',result)
    summarize_depth(argparse.Namespace(comparison=folder/'comparison.json',output=folder/'depth_residuals.json'))
    write(folder/'selected_audit_complete.json',dict(complete=True,selected_step=step,comparison_sha256=sha(folder/'comparison.json'),depth_residuals_sha256=sha(folder/'depth_residuals.json'),all103_converged=all(r['rgb_opacity_converged']for r in rows),source_model_mutated=False,launch_permission_granted=False))
    print(folder,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();main(a.campaign.resolve())
