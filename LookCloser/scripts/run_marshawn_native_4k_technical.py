"""Conditional technical-only native4K executor; waits for completed HD ownership."""
import argparse,copy,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import yaml
from yaml.nodes import ScalarNode,SequenceNode


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def alive(pid):
    try:return Path('/proc/%d/stat'%pid).read_text().split(') ')[1].split()[0]!='Z'
    except FileNotFoundError:return False


def memory():
    values={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.split(':')[0] in ['MemAvailable','MemTotal']};return values


def gpu():
    value=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip();return {int(line.split(',')[0]):int(line.split(',')[1]) for line in value.splitlines() if line}


def get(node,key):return next(value for name,value in node.value if name.value==key)


def put(node,key,value):
    for i,(name,prior) in enumerate(node.value):
        if name.value==key:node.value[i]=(name,value);return
    node.value.append((ScalarNode('tag:yaml.org,2002:str',key),value))


def scalar(value):return ScalarNode('tag:yaml.org,2002:str',str(value))
def pathnode(value):return SequenceNode('tag:yaml.org,2002:python/object/apply:pathlib.PosixPath',[scalar(value)])


def verify_bindings(bindings):
    for path,digest in bindings.items():
        if sha(path)!=digest:raise ValueError('Review-bound file changed: '+path)


def discover_source(production,hd_release):
    state=json.loads((production/'state.json').read_text());complete=json.loads((production/'complete.json').read_text());selected=state['selection'];assert complete['selection']==selected and state['plateau']['stopped']
    step=selected['step'];checkpoint=Path(selected['checkpoint']);assert sha(checkpoint)==selected['checkpoint_sha256']
    if step==20000:runtime=Path(hd_release['initial_runtime']);grid=Path(hd_release['initial_grid'])
    else:runtime=production/'runtime'/('step-%09d.json'%step);grid=production/'grids'/('step-%09d_grid.npz'%step)
    metadata=json.loads(grid.with_suffix('.json').read_text());assert metadata['checkpoint_sha256']==selected['checkpoint_sha256'] and sha(grid)==metadata['grid_archive_sha256']
    retained=[]
    for row in state['evaluations']:
        assert sha(row['checkpoint'])==row['checkpoint_sha256'];retained.append({k:row[k] for k in ['step','checkpoint','checkpoint_sha256']})
    return dict(step=step,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),runtime=str(runtime),runtime_sha256=sha(runtime),grid=str(grid),grid_sha256=sha(grid),grid_metadata_sha256=sha(grid.with_suffix('.json')),renders=selected['renders'],metrics=selected['metrics'],by_view=selected.get('by_view',{}),parent_state_sha256=sha(production/'state.json'),parent_complete_sha256=sha(production/'complete.json'),all_HD_checkpoint_controls=retained)


def compare_priors(reference,candidate,limits):
    import numpy as np
    left=json.loads((reference/'prior_probe.json').read_text());right=json.loads((candidate/'prior_probe.json').read_text());assert left['parameter_names']==right['parameter_names'] and left['parameter_shapes']==right['parameter_shapes'] and left['step']==right['step'];rows={}
    with np.load(reference/'prior_gradients.npz') as aa,np.load(candidate/'prior_gradients.npz') as bb:
        assert set(aa.files)==set(bb.files)
        for name in aa.files:
            a,b=aa[name],bb[name];assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all();passed=bool(np.allclose(a,b,rtol=limits['gradient_rtol'],atol=limits['gradient_atol']));assert passed,name
            rows[name]=dict(max_abs=float(np.max(np.abs(a-b))),relative_L2=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30)),passed=passed)
    for name,a in left['values'].items():assert np.isclose(a,right['values'][name],rtol=limits['value_rtol'],atol=limits['value_atol']),name
    return dict(reference=str(reference),candidate=str(candidate),gradients=rows,values_close=True,limits=limits)


def compare_render(source,folder,limits):
    import numpy as np
    from PIL import Image
    evaluation=json.loads((folder/'evaluation.json').read_text());old=source['metrics'];new=evaluation['v1']['metrics'];delta={k:new[k]-old[k] for k in ['psnr','ssim','lpips']};assert abs(delta['psnr'])<=limits['PSNR_absolute_dB'] and abs(delta['lpips'])<=limits['LPIPS_absolute']
    assert set(source['by_view'])=={'DC','HC','KA'}
    for camera,prior in source['by_view'].items():
        current=evaluation['v1']['by_view'][camera]
        assert abs(current['psnr']-prior['psnr'])<=limits['PSNR_absolute_dB'] and abs(current['lpips']-prior['lpips'])<=limits['LPIPS_absolute'],camera
    rows=[]
    for camera in ['DC','HC','KA']:
        original=Path(source['renders'])/camera;current=folder/'renders'/camera
        aa=json.loads((original/'receipt.json').read_text());bb=json.loads((current/'receipt.json').read_text())
        assert aa['camera_runtime']==bb['camera_runtime'] and aa['samples']==bb['samples'] and aa['box_sha256']==bb['box_sha256'] and aa['model_before_after_sha256']==bb['model_before_after_sha256']
        a=np.asarray(Image.open(original/(camera+'_inside.png'))).astype(np.int16);b=np.asarray(Image.open(current/(camera+'_inside.png'))).astype(np.int16);maximum=int(np.abs(a-b).max());assert maximum<=limits['PNG_max_code'],(camera,maximum)
        rows.append(dict(camera=camera,PNG_max_code=maximum,changed_RGB_values=int((a!=b).sum()),camera_samples_model_exact=True))
    return dict(metrics_delta=delta,views=rows,limits=limits)


def main(a):
    release=json.loads(a.release.read_text())
    if not release.get('technical_GPU_gate_authorized') or release.get('production_launch_authorized') is not False:raise ValueError('Reviewed technical-only conditional release required')
    if Path(sys.executable).resolve()!=Path(release['python']).resolve():raise ValueError('Use the reviewed historical Python environment')
    verify_bindings(release['bound_files']);hd_release=json.loads(Path(release['HD_root_release']).read_text());verify_bindings(hd_release['bound_files'])
    out=Path(release['output']);out.mkdir(parents=True,exist_ok=False);production=Path(release['HD_production']);plan=Path(release['reviewed_plan']);spec=json.loads((plan/'adapter/technical_probe_spec.json').read_text());launch=json.loads((production/'launch.json').read_text())
    atomic(out/'launch.json',dict(pid=os.getpid(),time=time.time(),release_sha256=sha(a.release),technical_only=True))
    lock=Path(release['GPU_lock']).open('a');locked=False
    while not locked:
        status=json.loads((production/'status.json').read_text());worker_alive=alive(status['pid']);controllers=[v for k,v in launch.items() if 'pid' in k.lower() and isinstance(v,int)];controller_alive=any(alive(v) for v in controllers);health=dict(time=time.time(),HD_status=status,HD_worker_alive=worker_alive,HD_controller_alive=controller_alive,HD_complete=(production/'complete.json').exists(),host=memory(),GPU=gpu())
        with (out/'wait_health.jsonl').open('a') as f:f.write(json.dumps(health)+'\n')
        if health['HD_complete'] and not worker_alive and not controller_alive and not health['GPU'] and health['host']['MemAvailable']>=release['minimum_available_RAM_GiB']*2**30:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locked=True
            except BlockingIOError:pass
        if not locked:time.sleep(30)
    assert not gpu() and memory()['MemAvailable']>=release['minimum_available_RAM_GiB']*2**30
    source=discover_source(production,hd_release);atomic(out/'selected_parent.json',source)
    phase_order=['original_HD_reference','HD_identity_zero','TRAIN4K_zero','TRAIN4K_updates16','TRAIN4K_resume16','TRAIN4K_final_roundtrip'];assert release['allowed_phases']==phase_order
    def run(command,label):
        if memory()['MemAvailable']<release['minimum_available_RAM_GiB']*2**30:raise RuntimeError('Insufficient available RAM before technical phase')
        with (out/(label+'.log')).open('w') as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                code=child.poll();mem=memory();usage=gpu();row=dict(time=time.time(),phase=label,pid=child.pid,returncode=code,host=mem,GPU=usage)
                with (out/'phase_health.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                if code is None and mem['MemAvailable']<release['abort_below_available_RAM_GiB']*2**30:
                    os.killpg(child.pid,signal.SIGTERM);child.wait();raise RuntimeError('RAM reserve breached; stopped own technical child')
                if code is not None:
                    if code:raise RuntimeError('Technical phase failed: '+label+' returncode='+str(code))
                    break
                time.sleep(10)
    (out/'configs').mkdir();(out/'jobs').mkdir();receipts={};parity={}
    for phase in phase_order:
        if phase=='TRAIN4K_updates16':assert set(parity)=={'HD_identity_zero_priors','HD_identity_zero_FHD','TRAIN4K_zero_priors','TRAIN4K_zero_FHD'}
        original=phase=='original_HD_reference';scale=1 if phase in ['original_HD_reference','HD_identity_zero'] else 2;template=Path(hd_release['config']) if original else Path(next(r['config'] for r in spec['configs'] if r['train_scale']==scale));configuration=yaml.compose(template.read_text());pipeline=get(configuration,'pipeline')
        parent=source
        if phase=='TRAIN4K_updates16':parent=receipts['TRAIN4K_zero']
        elif phase=='TRAIN4K_resume16':parent=receipts['TRAIN4K_updates16']
        elif phase=='TRAIN4K_final_roundtrip':parent=receipts['TRAIN4K_resume16']
        if 'output_checkpoint' in parent:parent=dict(checkpoint=parent['output_checkpoint'],checkpoint_sha256=parent['output_checkpoint_sha256'],runtime=parent['output_runtime'],runtime_sha256=parent['output_runtime_sha256'],grid=parent['output_grid'],grid_sha256=parent['output_grid_sha256'])
        put(configuration,'load_checkpoint',pathnode(parent['checkpoint']));put(configuration,'output_dir',pathnode(out/phase/'run'));put(configuration,'experiment_name',scalar('native_resolution_technical_'+phase));put(configuration,'timestamp',scalar('isolated'))
        put(pipeline,'final_grid_artifact_dir',pathnode(out/phase/'grids'))
        if not original:
            gate_source=Path(get(pipeline,'final_color_gate').value);gate=json.loads(gate_source.read_text());assert gate['approved'] is False;gate['approved']=True;gate['technical_root_release_sha256']=sha(a.release);gate['production_authorized']=False;gate_path=out/'configs'/(phase+'_RGB_gate.json');atomic(gate_path,gate);put(pipeline,'final_color_gate',scalar(gate_path));put(pipeline,'final_color_gate_sha256',scalar(sha(gate_path)))
        config_path=out/'configs'/(phase+'.yml');config_path.write_text(yaml.serialize(configuration))
        inputs={str(config_path):sha(config_path),parent['checkpoint']:parent['checkpoint_sha256'],parent['runtime']:parent['runtime_sha256'],parent['grid']:parent['grid_sha256'],str(Path(parent['grid']).with_suffix('.json')):sha(Path(parent['grid']).with_suffix('.json'))}
        job=dict(phase=phase,root_release=str(a.release),root_release_sha256=sha(a.release),bound_inputs=inputs,python_path=hd_release['python_path'] if original else spec['python_path'],config=str(config_path),source_checkpoint=parent['checkpoint'],source_runtime=parent['runtime'],source_grid=parent['grid'],output=str(out/phase),train_scale=scale,measure_priors=phase in phase_order[:3],successful_updates=16 if phase in ['TRAIN4K_updates16','TRAIN4K_resume16'] else 0,maximum_attempts=128)
        job_path=out/'jobs'/(phase+'.json');atomic(job_path,job);atomic(out/'status.json',dict(phase=phase,time=time.time(),parent_step=source['step']))
        run([sys.executable,release['worker'],'--job',str(job_path)],phase);receipt=json.loads((out/phase/'receipt.json').read_text());receipts[phase]=receipt
        if phase in ['HD_identity_zero','TRAIN4K_zero']:
            parity[phase+'_priors']=compare_priors(out/'original_HD_reference',out/phase,release['prior_parity_limits'])
            evaluation=out/(phase+'_FHD_evaluation');command=[sys.executable,hd_release['evaluator'],'--config',str(config_path),'--checkpoint',receipt['output_checkpoint'],'--checkpoint-sha256',receipt['output_checkpoint_sha256'],'--output',str(evaluation),'--ready',hd_release['ready'],'--box',hd_release['box'],'--v2-manifest',hd_release['v2_manifest'],'--renderer',hd_release['renderer']]
            for path in spec['python_path']:command+=['--python-path',path]
            run(command,phase+'_FHD_evaluation');parity[phase+'_FHD']=compare_render(source,evaluation,release['render_parity_limits']);atomic(out/'parity.json',parity)
        if phase=='TRAIN4K_updates16':assert set(parity)=={'HD_identity_zero_priors','HD_identity_zero_FHD','TRAIN4K_zero_priors','TRAIN4K_zero_FHD'}
    assert receipts['TRAIN4K_updates16']['actual_updates']==receipts['TRAIN4K_resume16']['actual_updates']==16
    assert receipts['TRAIN4K_resume16']['initial_adam']==receipts['TRAIN4K_updates16']['final_adam']
    assert receipts['TRAIN4K_final_roundtrip']['initial_pipeline_hash']==receipts['TRAIN4K_resume16']['final_pipeline_hash']
    verify_bindings(release['bound_files']);assert discover_source(production,hd_release)==source
    atomic(out/'complete.json',dict(time=time.time(),actual_Adam_updates=32,attempts=sum(receipts[p]['attempts'] for p in ['TRAIN4K_updates16','TRAIN4K_resume16']),source=source,parity=parity,phase_receipts={p:sha(out/p/'receipt.json') for p in phase_order},production_authorized=False,technical_checkpoints_excluded_from_production_ancestry=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release',type=Path,required=True);main(parser.parse_args())
