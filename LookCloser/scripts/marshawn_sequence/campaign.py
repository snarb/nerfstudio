"""Persistent six-hour two-station experiment controller and final-frame dispatcher.

Remote access uses an SSH key; credentials are never stored in campaign JSON.
"""
import argparse
import fcntl
import json
import math
import os
import shlex
import subprocess
import time
from pathlib import Path
from common import atomic,read,sha


ARMS=[
    ('uniform',dict(frequency_source_policy='uniform16')),
    ('uniform_gpu',dict(frequency_source_policy='uniform16',gpu_cache=True)),
    ('learned095',dict(frequency_source_policy='ssim095')),
    ('learned090',dict(frequency_source_policy='ssim090')),
    ('arm_half',dict(frequency_source_policy='uniform16',coarse_step=.00078125)),
    ('arm_double',dict(frequency_source_policy='uniform16',coarse_step=.003125)),
    ('arm_2048',dict(frequency_source_policy='uniform16',max_steps_per_ray=2048)),
    ('exr',dict(frequency_source_policy='uniform16',format='exr',coarse_step=.00078125)),
    ('resolution16384',dict(frequency_source_policy='uniform16',max_res=16384)),
    ('arm_max3',dict(frequency_source_policy='uniform16',adaptive_interval_level_mode='max3')),
    ('uniform_repeat',dict(frequency_source_policy='uniform16')),
    ('head20',dict(frequency_source_policy='uniform16',head_fraction=.20)),
    ('uniform_clever',dict(frequency_source_policy='uniform16')),
    ('fr_initial03',dict(frequency_source_policy='uniform16',fr_initial=.3)),
    ('frequency512',dict(frequency_source_policy='uniform16',frequency_grid_update_interval=512)),
    ('rig_units',dict(frequency_source_policy='uniform16',model_units_scale=1.)),
    ('batch16384',dict(frequency_source_policy='uniform16',train_rays_per_batch=16384)),
    ('learned095_arm_half',dict(frequency_source_policy='ssim095',coarse_step=.00078125)),
    ('confidence32',dict(frequency_source_policy='uniform16',confidence_width=32)),
    ('arm_quarter',dict(frequency_source_policy='uniform16',coarse_step=.000390625)),
]
_priority=['uniform','uniform_repeat','uniform_gpu','learned095','learned090','arm_half','arm_double','arm_2048','resolution16384','arm_max3','fr_initial03','frequency512','batch16384','head20','rig_units','exr','confidence32','learned095_arm_half','uniform_clever','arm_quarter']
ARMS.sort(key=lambda row:_priority.index(row[0]))


def make_spec(root,parameters,data,output,target,allow_plateau=False):
    root=Path(root);s=read(root/'settings.json');data=Path(data);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    parameters={'model_units_scale':s.get('model_units_scale',1.),**parameters}
    if abs(read(data/'ready.json').get('model_units_scale',1.)-parameters['model_units_scale'])>1e-10:raise ValueError('Dataset/spec model units differ')
    stage=output/'stage.json';atomic(stage,dict(stop_after_iterations=target))
    spec=dict(parameters,data=str(data),output=str(output),stage_plan=str(stage),source_code=s['source_code'],frozen_helpers=s['frozen_helpers'],nerfacc_extension=s['nerfacc_extension'],hard_max_iterations=150000,head_fraction=parameters.get('head_fraction',.15),head_proxy_manifest=str(data/'heads/manifest.json'),head_proxy_manifest_sha256=sha(data/'heads/manifest.json'),semantic_admission=dict(manifest=str(data/'admission/manifest.json'),manifest_sha256=sha(data/'admission/manifest.json')),gpu_cache=parameters.get('gpu_cache',s.get('gpu_cache',False)),allow_plateau=allow_plateau)
    if s.get('sampling_memmap_root'):
        spec['sampling_memmap']=str(Path(s['sampling_memmap_root'])/__import__('hashlib').sha256(str(output).encode()).hexdigest()[:16])
        spec['gpu_cache']=True
    # Only source bytes used by runtime/helpers are sealed; historical unavailable helpers are not fabricated.
    files=[Path(s['tools'])/name for name in ('common.py','runtime.py','worker.py')]+[p for p in Path(s['frozen_helpers']).glob('*.py')]
    files+=[p for p in Path(s['source_code']).rglob('*.py')]
    spec['code_sha256']={str(p):sha(p) for p in sorted(files)}
    path=output/'spec.json'
    if path.exists():
        if read(path)!=spec:raise ValueError('Immutable run spec changed '+str(path))
    else:atomic(path,spec)
    return path


class Host:
    def __init__(self,row):self.row=row;self.root=Path(row['root']);self.agent_root=Path(row.get('agent_root',row['root']))
    def argv(self,cmd):
        if self.row.get('run_as_user'):cmd='sudo -n -u '+shlex.quote(self.row['run_as_user'])+' bash -c '+shlex.quote(cmd)
        if self.row.get('local'):return ['bash','-c',cmd]
        args=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=accept-new']
        if self.row.get('key'):args+=['-i',self.row['key']]
        if self.row.get('via'):
            return args+[self.row['via'],'ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new '+shlex.quote(self.row['ssh'])+' '+shlex.quote(cmd)]
        return args+[self.row['ssh'],cmd]
    def command(self,cmd,input=None,timeout=60):
        r=subprocess.run(self.argv(cmd),input=input,capture_output=True,text=True,timeout=timeout)
        if r.returncode:raise RuntimeError(r.stderr[-1000:])
        return r.stdout
    def get(self,path,default=None):
        try:return json.loads(self.command('cat '+shlex.quote(str(path))))
        except (RuntimeError,json.JSONDecodeError):return default
    def put(self,path,value):
        content=json.dumps(value,indent=2)+'\n'
        command='python3 -c '+shlex.quote('import os,sys;from pathlib import Path;p=Path('+repr(str(path))+');p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(".controller.tmp");t.write_text(sys.stdin.read());os.replace(t,p)')
        self.command(command,input=content)
    def worker_job(self,name,parameters,target,round_name):
        settings=self.get(self.root/'settings.json')
        prefix='model_' if settings.get('model_units_scale',1.)!=1 else ''
        dataset=self.root/'variants'/(prefix+parameters['frequency_source_policy'])
        if parameters.get('model_units_scale')==1.:dataset=self.root/'variants'/'uniform16'
        if parameters.get('format')=='exr':dataset=self.root/'variants'/(prefix+'exr')
        if parameters.get('confidence_width')==32:
            parent=dataset;dataset=dataset.with_name(dataset.name+'_w32')
            build='from assets import confidence_variant;confidence_variant('+repr(str(parent))+','+repr(str(dataset))+',32)'
            self.command('PYTHONPATH='+shlex.quote(settings['tools']+':'+settings['dependency_path'])+' '+shlex.quote(settings['python'])+' -c '+shlex.quote(build),timeout=600)
        if parameters.get('max_res',8192)!=8192:
            source_dataset=dataset
            dataset=self.root/'variants'/(prefix+parameters['frequency_source_policy']+'_res'+str(parameters['max_res']))
            build='from assets import resolution_variant;resolution_variant('+repr(str(source_dataset))+','+repr(str(dataset))+','+repr(parameters['max_res'])+')'
            self.command('PYTHONPATH='+shlex.quote(settings['tools']+':'+settings['dependency_path'])+' '+shlex.quote(settings['python'])+' -c '+shlex.quote(build),timeout=120)
        output=self.root/'tuning'/name
        expression='from campaign import make_spec;make_spec('+repr(str(self.root))+','+repr(parameters)+','+repr(str(dataset))+','+repr(str(output))+','+repr(target)+')'
        self.command('PYTHONPATH='+shlex.quote(settings['tools'])+' '+shlex.quote(settings['python'])+' -c '+shlex.quote(expression))
        return dict(id=name+'_'+round_name,command=[settings['python'],str(Path(settings['tools'])/'worker.py'),'--spec',str(output/'spec.json')],completion=str(output/'stage_complete.json'),progress=str(output/'progress.json'),target=target,max_optimizer_attempts=150000,env={'PYTHONPATH':settings['dependency_path'],'CUDA_VISIBLE_DEVICES':str(self.row.get('gpu',0))})


def metrics(host,name):return host.get(host.root/'tuning'/name/'metrics.json',[])


def allocate_frames(hosts,first=908,last=1268):
    """Balance the whole sequence before reserving each local source range."""
    weights=[h.row.get('production_weight',1.) for h in hosts]
    if any(w<=0 for w in weights):raise ValueError('Positive worker weights required')
    total=last-first+1;expected=[total*w/sum(weights) for w in weights]
    quotas=[int(x) for x in expected]
    for i in sorted(range(len(hosts)),key=lambda i:expected[i]-quotas[i],reverse=True)[:total-sum(quotas)]:quotas[i]+=1
    remaining=set(range(first,last+1));allocation={i:[] for i in range(len(hosts))}
    for i,h in enumerate(hosts):
        if 'frame_range' not in h.row:continue
        lo,hi=h.row['frame_range'];available=sorted(f for f in remaining if lo<=f<=hi);count=quotas[i]
        if count>len(available):raise ValueError('Source range cannot meet worker quota')
        chosen=[available[round(k*(len(available)-1)/max(1,count-1))] for k in range(count)]
        allocation[i]=chosen;remaining.difference_update(chosen)
    for frame in sorted(remaining):
        eligible=[i for i,h in enumerate(hosts) if len(allocation[i])<quotas[i] and h.row.get('frame_range',[first,last])[0]<=frame<=h.row.get('frame_range',[first,last])[1]]
        if not eligible:raise ValueError('No source-capable production host for frame '+str(frame))
        index=min(eligible,key=lambda i:(len(allocation[i])/weights[i],i));allocation[index].append(frame)
    flattened=[f for frames in allocation.values() for f in frames]
    if sorted(flattened)!=list(range(first,last+1)):raise ValueError('Production coverage/uniqueness failure')
    return {i:sorted(frames) for i,frames in allocation.items()}


def run(root):
    root=Path(root)
    lease=(root/'campaign.lock').open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fleet=read(root/'fleet.json');hosts=[Host(h) for h in fleet['hosts']]
    state_path=root/'campaign_state.json'
    state=read(state_path) if state_path.exists() else dict(phase='prepared',assignments={},jobs={})
    if state['phase']=='prepared':
        if not all(h.get(h.root/'preflight_complete.json',{}).get('passed') for h in hosts):raise ValueError('All hosts need real preflight completion before tuning clock')
        state.update(phase='screen',started=time.time(),deadline=time.time()+21600);atomic(state_path,state)
    while state['phase'] in ('screen','extend'):
        if (root/'STOP').exists():
            for h in hosts:h.put(h.agent_root/'queue.json',dict(jobs=[]))
            atomic(state_path,dict(state,stopped=True));return
        now=time.time();statuses={i:h.get(h.agent_root/'agent_state.json',{}) for i,h in enumerate(hosts)}
        if now>=state['deadline']:
            for i,h in enumerate(hosts):
                h.put(h.agent_root/'queue.json',dict(jobs=[]))
                active=statuses[i].get('active')
                if active:
                    row=statuses[i]['jobs'][active]
                    # Signal only the process whose immutable identity the agent owns.
                    settings=h.get(h.root/'settings.json')
                    script='import sys,os,signal;sys.path.insert(0,'+repr(settings['tools'])+');from agent import identity;p='+repr(row['process'])+';os.kill(p["pid"],signal.SIGTERM) if identity(p["pid"])==p else None'
                    h.command('python3 -c '+shlex.quote(script))
            state['phase']='finalize';atomic(state_path,state);break
        if state['phase']=='screen':
            for i,h in enumerate(hosts):
                status=statuses[i]
                if status.get('active'):continue
                previous=[k for k,v in state['assignments'].items() if v==i]
                unseen=[name for name in previous if name+'_screen' not in status.get('jobs',{})]
                if unseen:
                    h.put(h.agent_root/'queue.json',dict(jobs=[state['jobs'][name] for name in unseen]));continue
                if any(status.get('jobs',{}).get(name+'_screen',{}).get('status') not in ('complete','failed') for name in previous):continue
                pending=[(n,p) for n,p in ARMS if n not in state['assignments'] and (p.get('format','png') in h.row.get('formats',['png'])) and (n!='arm_quarter' or h.row['name']=='ws07')]
                if not pending:
                    if str(i) not in state.get('early_extensions',{}):
                        scored=[(r['metrics']['lpips'],name) for name in previous for r in metrics(h,name) if r['step']+1==32000 and r['cap']==10]
                        if scored:
                            _,name=min(scored);job=h.worker_job(name,dict(ARMS)[name],150000,'early150000')
                            state.setdefault('early_extensions',{})[str(i)]=name;state['jobs'][name+'_early150000']=job;atomic(state_path,state)
                            h.put(h.agent_root/'queue.json',dict(jobs=[job]))
                    continue
                # Reserve at least2h for longer trajectories; never silently drop scientific comparisons.
                if now-state['started']>4*3600:continue
                name,parameters=pending[0];job=h.worker_job(name,parameters,32000,'screen')
                state['assignments'][name]=i;state['jobs'][name]=job;atomic(state_path,state)
                h.put(h.agent_root/'queue.json',dict(jobs=[job]))
            exhausted=len(state['assignments'])==len(ARMS) and all(not s.get('active') for s in statuses.values())
            if exhausted or now-state['started']>=4*3600:
                if any(s.get('active') for s in statuses.values()):time.sleep(30);continue
                candidates=[]
                for name,index in state['assignments'].items():
                    rows=[r for r in metrics(hosts[index],name) if r['step']+1==32000 and r['cap']==10]
                    if rows:candidates.append((rows[0]['metrics']['lpips'],name,index))
                if not candidates:raise ValueError('No matched32k candidate; do not launch production')
                candidates.sort();used=set()
                for _,name,index in candidates:
                    if index in used:continue
                    h=hosts[index];used.add(index);p=dict(dict(ARMS)[name]);progress=h.get(h.root/'tuning'/name/'progress.json',{})
                    seconds_per_step=progress.get('elapsed_seconds',3600)/max(1,progress.get('completed_iterations',32000))
                    feasible=32000+int((state['deadline']-now)*.65/max(.001,seconds_per_step))
                    target=min(150000,max(32000,16000*(feasible//16000)))
                    job=h.worker_job(name,p,target,'extend');h.put(h.agent_root/'queue.json',dict(jobs=[job]));state['jobs'][name+'_extend']=job
                state['phase']='extend';atomic(state_path,state)
        elif all(not s.get('active') for s in statuses.values()):
            # A completed stage may leave budget for one more16k continuation of each host's leader.
            for i,h in enumerate(hosts):
                names=[name for name,index in state['assignments'].items() if index==i]
                candidates=[(r['metrics']['lpips'],n,r) for n in names for r in metrics(h,n) if r['cap']==10]
                if not candidates:continue
                _,name,best=min(candidates);progress=h.get(h.root/'tuning'/name/'progress.json',{})
                done=progress.get('completed_iterations',32000);sps=progress.get('elapsed_seconds',3600)/max(1,done)
                if done<150000 and state['deadline']-now>16000*sps*1.5:
                    target=min(150000,done+16000);job=h.worker_job(name,dict(ARMS)[name],target,'extend'+str(target));h.put(h.agent_root/'queue.json',dict(jobs=[job]))
        with (root/'controller_checks.jsonl').open('a') as stream:stream.write(json.dumps(dict(time=now,phase=state['phase'],agents=statuses))+'\n')
        time.sleep(30)
    if state['phase']=='finalize':
        # Let any checkpoint/evaluation already in progress finish before reading committed metrics.
        for _ in range(40):
            if all(not h.get(h.agent_root/'agent_state.json',{}).get('active') for h in hosts):break
            time.sleep(15)
        candidates=[]
        for name,index in state['assignments'].items():
            for row in metrics(hosts[index],name):
                if row['cap'] in (None,10):candidates.append(dict(name=name,host=index,result=row,parameters=dict(ARMS)[name]))
        if not candidates:raise ValueError('No fully committed candidate')
        png=[c for c in candidates if c['parameters'].get('format')!='exr']
        baseline=min(png,key=lambda c:c['result']['metrics']['lpips'])
        # EXR additionally needs detail evidence; absent evidence defaults honestly to PNG.
        def exr_qualifies(c):
            if c['parameters'].get('format')!='exr':return True
            def training_key(row):return {k:v for k,v in row['parameters'].items() if k not in ('format','gpu_cache')}
            pairs=[p for p in png if training_key(p)==training_key(c) and p['result']['step']==c['result']['step'] and p['result']['cap']==c['result']['cap']]
            if not pairs:
                c['exr_detail_adoption_check']=dict(passed=False,reason='No matched PNG parameter/step/cap gate');return False
            reference=min(pairs,key=lambda p:p['result']['metrics']['lpips'])
            a=c['result'].get('head_lpips_by_view',{});b=reference['result'].get('head_lpips_by_view',{})
            shared=set(a)&set(b)
            passed=c['result']['metrics']['lpips']<reference['result']['metrics']['lpips']*.98 and sum(a[v]<b[v]*.98 for v in shared)>=2 and all(a[v]<=b[v]*1.05 for v in shared)
            c['exr_detail_adoption_check']=dict(passed=passed,matched_png_arm=reference['name'],matched_png_step=reference['result']['step']+1,matched_png_cap=reference['result']['cap'],shared_views=sorted(shared),policy='same hyperparameters/steps/cap; >=2% actor LPIPS improvement; >=2% head detail improvement in at least2views; no head view regresses>5%; common SDR reference')
            return passed
        eligible=[c for c in candidates if exr_qualifies(c)]
        best_lpips=min(c['result']['metrics']['lpips'] for c in eligible)
        tied=[c for c in eligible if c['result']['metrics']['lpips']<=best_lpips*1.005]
        absolute_best=min(eligible,key=lambda c:c['result']['metrics']['lpips'])
        def reasonable_detail(c):
            a=c['result'].get('head_lpips_by_view',{});b=absolute_best['result'].get('head_lpips_by_view',{})
            return all(a.get(v,float('inf'))<=value*1.05 for v,value in b.items())
        tied=[c for c in tied if reasonable_detail(c)] or [absolute_best]
        def estimated_training_seconds(c):
            p=hosts[c['host']].get(hosts[c['host']].root/'tuning'/c['name']/'progress.json',{})
            return (c['result']['step']+1)*p.get('elapsed_seconds',3600)/max(1,p.get('completed_iterations',32000))
        chosen=min(tied,key=lambda c:(estimated_training_seconds(c),c['result']['metrics']['lpips']))
        recipe=dict(parameters=dict(chosen['parameters']),steps=chosen['result']['step']+1,selection=chosen,all_candidates=candidates,primary='equal visible-person masked AlexLPIPS then equal heldout views',plateau_confirmed=False,limit=150000,source_frame=908,final_train_cameras=72)
        recipe['parameters'].setdefault('model_units_scale',read(root/'settings.json').get('model_units_scale',1.))
        canonical=read(root/'box.json')
        recipe['canonical_geometry']={key:canonical[key] for key in ('aabb','world_to_box','box_to_world') if key in canonical}
        recipe['canonical_aabb_sha256']=sha(root/'box.json')
        recipe['rig_units_per_metre']=1.0805
        recipe['physical_box_dimensions_metres']=[(hi-lo)/1.0805 for lo,hi in zip(canonical['aabb'][0],canonical['aabb'][1])]
        recipe['tuning_heldout']=read(root/'split.json')['heldout']
        recipe['tuning_started']=state['started'];recipe['tuning_deadline']=state['deadline']
        trajectory=hosts[chosen['host']].get(hosts[chosen['host']].root/'tuning'/chosen['name']/'stage_complete.json',{})
        recipe['plateau_confirmed']=bool(trajectory.get('plateau',False))
        recipe['trajectory_completed_iterations']=trajectory.get('completed_iterations')
        recipe['selected_inference_cap']=chosen['result']['cap']
        recipe['selection_tolerance']=dict(relative_actor_lpips=.005,max_head_view_regression=.05,tie_break='estimated training wall time')
        # Gather changes are mathematically equivalent (contract-tested), but memory
        # use alone does not justify deployment. Compare matched hardware/cold runs.
        cache_check=dict(adopted=False,minimum_speedup=.05,reason='No completed matched 32k cache pair')
        if all(n in state['assignments'] for n in ('uniform_repeat','uniform_gpu')):
            pair={}
            for name in ('uniform_repeat','uniform_gpu'):
                h=hosts[state['assignments'][name]]
                job=h.get(h.agent_root/'agent_state.json',{}).get('jobs',{}).get(name+'_screen',{})
                matched=[m for m in metrics(h,name) if m['step']+1==32000 and m['cap']==10]
                if matched and job.get('status')=='complete' and job.get('finished',0)>job.get('started',0):
                    pair[name]=(job['finished']-job['started'])/32000
            if len(pair)==2:
                speedup=pair['uniform_repeat']/pair['uniform_gpu']-1
                cache_check.update(adopted=speedup>=.05,speedup=speedup,seconds_per_step=pair,reason='Matched cold native runs on Silver; includes checkpoint/evaluation overhead')
        recipe['gpu_cache_benchmark']=cache_check
        recipe['parameters']['gpu_cache']=cache_check['adopted']
        p=recipe['parameters']
        recipe['effective_hyperparameters']=dict(hash_levels=16,hash_features_per_level=2,log2_hashmap_size=25,max_res=p.get('max_res',8192),sh_degree=4,correct_sh_directions=True,density_activation='trunc_exp',train_rays_per_batch=p.get('train_rays_per_batch',8192),sampling='body80 with exact rounded proposal expectation, no 1/q correction',head_fraction=p.get('head_fraction',.15),confidence_width_native=p.get('confidence_width',64),opaque_core_erosion_native_pixels=32,opaque_core_prior=.01,adam_betas=[.9,.99],learning_rate=.01,learning_rate_final=.0001,learning_rate_horizon=200000,warmup_steps=4096,distortion_weight_start=.01,distortion_weight_final=.1,distortion_ramp=[4096,12288],fr_initial=p.get('fr_initial',1.),fr_switch_step=75941,fr_after_switch=.3,coarse_step=p.get('coarse_step',.0015625),minimum_step=.0001,maximum_step=.1,max_samples_per_ray=p.get('max_steps_per_ray',1024),corrected_arm_allocator=True,interval_level_mode=p.get('adaptive_interval_level_mode','midpoint'),occupancy_resolution=128,occupancy_ema=.95,occupancy_threshold=.01,occupancy_update_interval=16,occupancy_dilation_radius=0,frequency_grid_update_interval=p.get('frequency_grid_update_interval',1024),frequency_projection='legacy',frequency_probe_train_rays=2048,frequency_source_policy=p['frequency_source_policy'],native_rgb_resize=False,seed=42)
        for h in hosts:h.put(h.root/'selected_recipe.json',recipe)
        atomic(root/'selected_recipe.json',recipe)
        text='# Marshawn908–1268 selected native recipe\n\nPrimary: masked actor AlexLPIPS, without background.\n\nSelected parameters:\n```json\n'+json.dumps(recipe['parameters'],indent=2)+'\n```\n\nSteps: '+str(recipe['steps'])+'. Native crops without resize; accepted V3 box.\n\n|Arm|Host|Step|LPIPS|PSNR|SSIM|\n|---|---|---:|---:|---:|---:|\n'
        for c in candidates:
            m=c['result']['metrics'];text+='|%s|%s|%s|%.6f|%.4f|%.4f|\n'%(c['name'],c['host'],c['result']['step']+1,m['lpips'],m['psnr'],m['ssim'])
        text+='\nAll final models cold-start on72TRAIN cameras; metrics on former heldout cameras are in-sample. Maximum150000attempts including recovery replay. FPS120 remains assumed. Plateau is only asserted if the saved late trajectory passes the two-signal test. EXR sources are decoded from PQ16, not independent higher-information originals.\n'
        text+='\nInference cap: '+str(recipe['selected_inference_cap'])+'. Plateau confirmed: '+str(recipe['plateau_confirmed'])+'. Trajectory completed iterations: '+str(recipe['trajectory_completed_iterations'])+'.\n'
        text+='\nEffective training settings:\n```json\n'+json.dumps(recipe['effective_hyperparameters'],indent=2)+'\n```\n'
        text+='\nFixed tuning heldouts: '+', '.join(recipe['tuning_heldout'])+'. Canonical physical rig geometry (apply world_to_box before testing aabb; then apply the isotropic model-unit scale above):\n```json\n'+json.dumps(recipe['canonical_geometry'],indent=2)+'\n```\n'
        text+='\nRig calibration: 1.0805units/metre. Physical box dimensions: '+', '.join('%.4fm'%d for d in recipe['physical_box_dimensions_metres'])+'.\n'
        text+='\nGPU image cache benchmark:\n```json\n'+json.dumps(cache_check,indent=2)+'\n```\n'
        exr_checks=[c.get('exr_detail_adoption_check') for c in candidates if c.get('exr_detail_adoption_check')]
        text+='\nEXR detail adoption checks passed: '+str(sum(c['passed'] for c in exr_checks))+' / '+str(len(exr_checks))+'. Full head/view evidence and fixed-reference policy are recorded in selected_recipe.json.\n'
        (root/'selected_recipe.md').write_text(text)
        for host in hosts:
            if host.row['name']=='ws07':
                destination='/home/pavlo/nerfstudio/LookCloser/experiments/marshawn_908_1268_tuned_recipe.md'
                command='python3 -c '+shlex.quote('import sys;from pathlib import Path;Path('+repr(destination)+').write_text(sys.stdin.read())')
                host.command(command,input=text)
        state['phase']='production';atomic(state_path,state)
    if state['phase']=='production':
        allocation=allocate_frames(hosts)
        state['production_allocation']={hosts[i].row['name']:frames for i,frames in allocation.items()};atomic(state_path,state)
        for i,h in enumerate(hosts):
            s=h.get(h.root/'settings.json');frames=allocation[i]
            jobs=[dict(id='frame%06d'%f,command=[s['python'],str(Path(s['tools'])/'production.py'),'--root',str(h.root),'--frame',str(f)],completion=str(h.root/'frames'/f'{f:06d}'/'frame_complete.json'),progress=str(h.root/'frames'/f'{f:06d}'/'train/progress.json'),max_attempts=12,retry_backoff_seconds=60,env={'PYTHONPATH':s['dependency_path'],'CUDA_VISIBLE_DEVICES':str(h.row.get('gpu',0))}) for f in frames]
            h.put(h.agent_root/'queue.json',dict(jobs=jobs))
        state['phase']='production_dispatched';atomic(state_path,state)
    # Controller remains an independent watcher until every assigned frame finalizes/quarantines.
    while state['phase']=='production_dispatched':
        checks=[h.get(h.agent_root/'agent_state.json',{}) for h in hosts]
        completed=sum(sum(1 for k,v in s.get('jobs',{}).items() if k.startswith('frame') and v['status']=='complete') for s in checks)
        quarantined=sum(sum(1 for k,v in s.get('jobs',{}).items() if k.startswith('frame') and v['status']=='failed' and v['attempts']>=v['job'].get('max_attempts',3)) for s in checks)
        atomic(root/'sequence_progress.json',dict(time=time.time(),completed=completed,quarantined=quarantined,total=361))
        if completed+quarantined==361:
            state.update(phase='complete' if not quarantined else 'complete_with_quarantine',completed=completed,quarantined=quarantined);atomic(state_path,state);return
        time.sleep(60)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
