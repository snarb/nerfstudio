#!/usr/bin/env python3
"""Independent, CPU-only completed-result reporter. Never edits training state.

Packages actual renders from completed jobs only; checkpoint bytes are verified
against their source inventory but are not included in ZIP downloads. Numerical
ranking is provisional until a human/agent actually reviews the images.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, shlex, shutil, subprocess, sys, time, zipfile
from pathlib import Path
import fcntl


def read(path, default=None):
    try: return json.loads(Path(path).read_text())
    except FileNotFoundError: return default


def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.%d.tmp'%os.getpid())
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def identity(pid=None):
    pid=os.getpid() if pid is None else int(pid)
    try:
        s=Path('/proc/%d/stat'%pid).read_text().split(') ')[1].split()
        return dict(pid=pid,start=s[19],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),state=s[0])
    except FileNotFoundError:return None


def alive(record):
    current=identity(record['pid']) if record else None
    return bool(current and current['state']!='Z' and all(current[k]==record[k] for k in ['pid','start','boot']))


def scores(result):
    s=result['selection'];v=s.get('metrics',s)
    out={k:float(v.get(k,s.get('masked_lpips') if k=='lpips' else s.get(k))) for k in ['psnr','ssim','lpips']}
    if not all(math.isfinite(x) for x in out.values()):raise ValueError('Nonfinite completed quality score')
    return out


def choose(rows):
    if not rows:return None
    best=min(r['scores']['lpips'] for r in rows)
    return min((r for r in rows if r['scores']['lpips']<=best+.001),key=lambda r:(-r['scores']['psnr'],r['step'],r['name']))


def guard(candidate,baseline):
    if baseline is None:return dict(status='unknown',reason='Control has not completed')
    try:
        differences={}
        for actor in ['black','grey']:
            value=float(candidate['selection']['by_actor'][actor]['lpips'])
            ref=float(baseline['selection']['by_actor'][actor]['lpips'])
            if not math.isfinite(value) or not math.isfinite(ref) or min(value,ref)<0:raise ValueError('invalid actor LPIPS')
            differences[actor]=dict(lpips=value,control_lpips=ref,pass_guard=value<=ref*1.03)
        return dict(status='pass' if all(v['pass_guard'] for v in differences.values()) else 'fail',actors=differences)
    except (KeyError,TypeError,ValueError) as exc:return dict(status='unknown',reason='Missing/invalid actor scores: '+str(exc))


def is_selector(job):
    command=job.get('command',[])
    try:return bool(json.loads(command[command.index('--task')+1]).get('select_only'))
    except (ValueError,IndexError,TypeError):return False


def assemble(plan,state):
    specs={j['name']:j for j in plan['jobs']}; rows=[];completed=[];repeats=[]
    baseline=None
    for name,item in state['jobs'].items():
        check=item.get('latest_check',{});progress=list(check.get('progress',{}).values())
        progress=[p for p in progress if isinstance(p,dict)]
        step=max((p.get('step',-1) for p in progress),default=None)
        row=dict(name=name,status=item['status'],host=item.get('host'),gpu=item.get('gpu'),step=step,
                 heartbeat=check.get('time'),oom_evidence=check.get('oom_evidence'),scores=None)
        result=item.get('metrics')
        if item['status']=='complete' and result and not result.get('skipped') and 'selection' in result:
            row.update(scores=scores(result),step=int(result['selection']['step']))
            candidate=dict(name=name,host=item.get('host'),step=row['step'],scores=row['scores'],result=result)
            if result.get('technical_smoke'):row['classification']='technical_smoke_excluded'
            elif is_selector(specs.get(name,{})):row['classification']='selector_excluded'
            elif result.get('config',{}).get('seed')==42:
                completed.append(candidate)
                if name=='full_baseline_r0':baseline=candidate
            else:repeats.append(candidate);row['classification']='repeat_confirmation'
        if result and result.get('skipped'):row['status']='skipped'
        rows.append(row)
    for candidate in completed:candidate['guard']=guard(candidate['result'],baseline['result'] if baseline else None)
    raw=choose(completed);safe=choose([r for r in completed if r['guard']['status']=='pass'])
    status='awaiting_completed_control_and_candidates' if raw is None or baseline is None else (
           'provisional_manual_review_pending' if raw['guard']['status']=='pass' else 'automatic_promotion_withheld_actor_regression')
    primary=raw if raw and raw['guard']['status']=='pass' else None
    def compact(c):return {k:v for k,v in c.items() if k!='result'} if c else None
    confirmation=[]
    for repeat in repeats:
        ref=baseline if repeat['name']=='repeat_control_seed43' else raw
        confirmation.append(dict(name=repeat['name'],scores=repeat['scores'],reference=ref['name'] if ref else None,
             deltas={k:repeat['scores'][k]-ref['scores'][k] for k in ['psnr','ssim','lpips']} if ref else None,
             paired_seed=repeat['result'].get('config',{}).get('seed'),interpretation='Confirmation only; never best-of-seeds selection'))
    report=dict(time=time.time(),status=status,visual_review_complete=False,jobs=rows,raw_best=compact(raw),
                best_guard_passing=compact(safe),primary=compact(primary),baseline=compact(baseline),repeat_confirmation=confirmation,
                score_domain='FHD fixed actor masks/crops; identical grey exterior; equal actors and views',
                selection_rule='minimum masked LPIPS; .001 tie window then highest actor PSNR then earlier step',
                complete_only=True,controller_updated=state.get('updated'))
    return report,{c['name']:c for c in [raw,safe,baseline] if c}


SOURCE_PROBE=r'''
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);step=int(sys.argv[2]); result=json.loads((p/'result.json').read_text()); inv=json.loads((p/'checkpoint_inventory.json').read_text());record=inv[str(step)]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as s:
  for b in iter(lambda:s.read(8<<20),b''):h.update(b)
 return h.hexdigest()
checkpoint=p/record['path']; actual=sha(checkpoint)
assert record['retained'] and actual==record['sha256'], 'Selected checkpoint missing/changed'
assert result['selected_step']==step and pathlib.Path(result['checkpoint'])==checkpoint,'Result/checkpoint mismatch'
folder=p/'renders'/('step-%09d'%step)
assert pathlib.Path(result['render_directory'])==folder
metrics=json.loads((folder/'metrics.json').read_text()); assert metrics['step']==step
expected=['DC.png','HC.png','KA.png','DC_alpha.png','HC_alpha.png','KA_alpha.png','metrics.json']
files={name:{'sha256':sha(folder/name),'bytes':(folder/name).stat().st_size} for name in expected}
print(json.dumps(dict(checkpoint=dict(path=str(checkpoint),sha256=actual,bytes=checkpoint.stat().st_size),render_directory=str(folder),files=files,result=result,metrics=metrics,complete=json.loads((p/'complete.json').read_text()))))
'''


def source_probe(host,output,step):
    path=Path(output);root=Path(host['root'])
    if '..' in path.parts or not path.is_absolute() or not path.is_relative_to(root):raise ValueError('Output outside declared host root')
    argv=[host['python'],'-c',SOURCE_PROBE,str(path),str(step)]
    if host.get('address'):argv=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host['address'],shlex.join(argv)]
    result=subprocess.run(argv,capture_output=True,text=True,timeout=180)
    if result.returncode:raise RuntimeError(result.stderr[-2000:])
    return json.loads(result.stdout)


def copy_render_file(host,source,dest):
    if host.get('address'):
        # rsync protects remote argument bytes; source paths are trusted receipts.
        subprocess.run(['rsync','-a','--protect-args','-e','ssh -o BatchMode=yes -o ConnectTimeout=8',host['address']+':'+source,str(dest)],check=True,capture_output=True,timeout=180)
    else:shutil.copy2(source,dest)


def package(root,plan,report,candidates):
    if not report['raw_best'] or not report['baseline']:return None
    key=fingerprint({k:report[k] for k in ['raw_best','best_guard_passing','baseline','status']})[:16]
    downloads=Path(root)/'downloads';downloads.mkdir(exist_ok=True)
    latest=read(downloads/'latest.json',{})
    if latest.get('key')==key and Path(latest.get('zip','')).is_file() and sha(latest['zip'])==latest.get('zip_sha256'):return latest
    bundle=downloads/('review_'+key);bundle.mkdir(exist_ok=True)
    receipts={}
    for name,candidate in candidates.items():
        host=plan['hosts'][candidate['host']];output=candidate['result']['config']['output']
        source=source_probe(host,output,candidate['step'])
        if scores(source['complete'])!=candidate['scores']:raise ValueError('Controller/source metrics disagree: '+name)
        if fingerprint(source['metrics'])!=fingerprint(candidate['result']['selection']):raise ValueError('Render metrics do not match selected result')
        folder=bundle/name;folder.mkdir(exist_ok=True)
        for file,metadata in source['files'].items():
            dest=folder/file
            if not dest.exists() or sha(dest)!=metadata['sha256']:
                copy_render_file(host,str(Path(source['render_directory'])/file),dest)
            if sha(dest)!=metadata['sha256']:raise ValueError('Copied render SHA mismatch')
        atomic(folder/'config.json',candidate['result']['config']);atomic(folder/'checkpoint_provenance.json',dict(host=host.get('address') or 'local',**source['checkpoint']))
        receipts[name]=dict(host=host.get('address') or 'local',checkpoint=source['checkpoint'],renders=source['files'])
    shutil.copy2(Path(root)/'data/box.json',bundle/'box.json')
    atomic(bundle/'selection.json',report);atomic(bundle/'artifact_receipt.json',dict(verified=time.time(),files=receipts,box_sha256=sha(bundle/'box.json'),visual_review_complete=False))
    raw=report['raw_best'];safe=report['best_guard_passing']
    text='Actual completed-checkpoint renders; no generated/repainted pixels.\n\n'
    text+='Status: '+report['status']+'\nManual visual review remains pending.\n'
    text+='Raw best: '+raw['name']+' step '+str(raw['step'])+'\n'
    text+='Best passing per-actor regression guard: '+(safe['name'] if safe else 'none')+'\n'
    text+='Control: full_baseline_r0. Compare DC.png, HC.png, KA.png in each folder.\n'
    text+='Metric: FHD masked AlexNet LPIPS, fixed GT actor masks/crops with identical RGB0.5 exterior; equal actors and views.\n'
    text+='Checkpoint files are not in this ZIP; their source host/path/SHA256 are recorded per folder.\n'
    text+='Grey actor and black actor LPIPS must each remain within3% of the control for automatic numerical eligibility.\n'
    (bundle/'README.txt').write_text(text)
    archive=downloads/('marshawn_actor_review_'+key+'.zip');temporary=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        for file in sorted(bundle.rglob('*')):
            if file.is_file():z.write(file,str(file.relative_to(bundle)))
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:raise ValueError('ZIP integrity failure')
    os.replace(temporary,archive)
    receipt=dict(key=key,zip=str(archive),zip_sha256=sha(archive),created=time.time(),status=report['status'],visual_review_complete=False,
                 scp='scp coder@olive-lark:'+shlex.quote(str(archive))+' ~/Downloads/')
    atomic(downloads/'latest.json',receipt);return receipt


def markdown(report):
    lines=['# Marshawn1660 tuning','',report['status'],'',
           'Controller alive: '+str(report.get('controller_alive','unknown'))+'; state age: '+str(round(report.get('controller_state_age_seconds',0)))+' seconds.','',
           'Manual visual review: **pending**. Completed results only; no online checkpoint is promoted.','',
           '| Job | State | Host/GPU | Step | PSNR | SSIM | Masked LPIPS | OOM evidence |','|---|---|---|---:|---:|---:|---:|---|']
    for row in report['jobs']:
        s=row['scores'] or {};values=['%.6f'%s[k] if k in s else '—' for k in ['psnr','ssim','lpips']]
        oom='yes' if row['oom_evidence'] is True else 'none in latest check' if row['oom_evidence'] is False else 'unknown'
        lines.append('| '+ ' | '.join([row['name'],row['status'],str(row['host'])+'/'+str(row['gpu']),str(row['step']),*values,oom])+' |')
    lines+=['','Primary numerical candidate: '+(report['primary']['name'] if report['primary'] else 'withheld / not available')+'.',
            'Raw best: '+(report['raw_best']['name'] if report['raw_best'] else 'not available')+'.',
            'Best passing actor regression guard: '+(report['best_guard_passing']['name'] if report['best_guard_passing'] else 'not available')+'.']
    if report.get('download'):lines+=['','Download: `'+report['download']['scp']+'`']
    if report.get('package_error'):lines+=['','Artifact packaging incomplete: '+report['package_error']]
    for repeat in report['repeat_confirmation']:lines+=['',repeat['name']+' confirmation: '+json.dumps(repeat['deltas'])+' (PSNR/SSIM/LPIPS delta; never used for best-of-seed selection).']
    lines+=['','Metrics are specific to this fixed masked protocol, not historical full-frame LPIPS.','']
    return '\n'.join(lines)


def once(root,controller,*,do_package=True):
    root=Path(root);controller=Path(controller);plan=read(controller/'manifest.json');state=read(controller/'state.json')
    report,candidates=assemble(plan,state)
    control=read(controller/'controller.json',{})
    report['controller_alive']=alive(control.get('process'));report['controller_state_age_seconds']=max(0,time.time()-state.get('updated',0))
    if do_package:
        try:report['download']=package(root,plan,report,candidates)
        except Exception as exc:report['package_error']=str(exc)
    atomic(root/'report.json',report)
    tmp=root/'report.md.tmp';tmp.write_text(markdown(report));os.replace(tmp,root/'report.md')
    return report


def watch(root,controller,interval=60):
    directory=Path(root)/'reporter';directory.mkdir(exist_ok=True)
    held=(directory/'watcher.lock').open('a+')
    try:fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    status=dict(process=identity(),started=time.time(),status='running');atomic(directory/'status.json',status)
    while True:
        try:
            report=once(root,controller)
            status.update(heartbeat=time.time(),report_status=report['status'],package_error=report.get('package_error'))
            terminal=all(r['status'] in ('complete','failed','blocked','skipped') for r in report['jobs'])
            if terminal and not report.get('package_error'):status['status']='complete'
        except Exception as exc:status.update(heartbeat=time.time(),last_error=repr(exc));terminal=False
        atomic(directory/'status.json',status)
        if terminal and status['status']=='complete':return
        time.sleep(interval)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['once','launch','watch','status']);p.add_argument('--campaign-root',type=Path,required=True);p.add_argument('--controller-root',type=Path);p.add_argument('--interval',type=float,default=60)
    a=p.parse_args();root=a.campaign_root.resolve();controller=(a.controller_root or root/'controller').resolve();directory=root/'reporter';directory.mkdir(exist_ok=True)
    if not 1<=a.interval<=60:raise ValueError('Interval must be1..60 seconds')
    if a.action=='once':print(json.dumps({'status':once(root,controller)['status']}));return
    if a.action=='watch':watch(root,controller,a.interval);return
    status=read(directory/'status.json',{})
    if a.action=='status':print(json.dumps(dict(status,alive=alive(status.get('process'))),indent=2));return
    if alive(status.get('process')):print('Watcher already running');return
    frozen=directory/'report_marshawn_tuning.py'
    if frozen.exists() and sha(frozen)!=sha(__file__):raise ValueError('Existing watcher code differs; use a new reporter generation')
    if not frozen.exists():shutil.copy2(__file__,frozen)
    with (directory/'watcher.log').open('ab') as stream:
        process=subprocess.Popen([sys.executable,str(frozen),'watch','--campaign-root',str(root),'--controller-root',str(controller),'--interval',str(a.interval)],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps({'watcher_pid':process.pid,'report':str(root/'report.md')}))

if __name__=='__main__':main()
