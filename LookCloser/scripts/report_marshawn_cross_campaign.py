#!/usr/bin/env python3
"""Read-only cross-campaign CPU reporter; numerical ranking never grants approval."""
from __future__ import annotations
import argparse, fcntl, hashlib, json, math, os, shlex, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DOMAIN = 'fixed GT actor crop; identical RGB0.5 exterior; AlexNet LPIPS without resize'
PROTOCOL = 'v1_fixed_actor_mask_grey_exterior_equal_actor_camera'
AGGREGATION = 'equal actors per camera, then equal cameras'


def digest(raw): return hashlib.sha256(raw).hexdigest()


def canonical(value): return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.%d.tmp'%os.getpid())
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def identity(pid=None):
    pid=os.getpid() if pid is None else pid
    try:
        stat=Path('/proc/%d/stat'%pid).read_text().split(') ')[1].split()
        return dict(pid=pid,start=stat[19],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),state=stat[0])
    except FileNotFoundError:return None


def alive(record):
    current=identity(record['pid']) if record else None
    return bool(current and current['state']!='Z' and all(current[k]==record[k] for k in ['pid','start','boot']))


PROBE = r'''
import json,hashlib,pathlib,sys,time
out={}
for key,path in json.load(sys.stdin).items():
 try:
  p=pathlib.Path(path);raw=p.read_bytes();stat=p.stat();json.loads(raw)
  out[key]={'raw':raw.decode(),'sha256':hashlib.sha256(raw).hexdigest(),'mtime':stat.st_mtime,'read_at':time.time(),'path':str(p)}
 except Exception as exc:out[key]={'error':type(exc).__name__+': '+str(exc),'path':path,'read_at':time.time()}
print(json.dumps(out))
'''


def fetch(host, paths):
    argv=[host.get('python',sys.executable),'-c',PROBE]
    if host.get('address'):
        argv=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host['address'],shlex.join(argv)]
    result=subprocess.run(argv,input=json.dumps(paths),capture_output=True,text=True,timeout=30)
    if result.returncode:raise RuntimeError(result.stderr[-1000:])
    records=json.loads(result.stdout)
    for record in records.values():
        if 'raw' in record:
            if digest(record['raw'].encode())!=record['sha256']:raise ValueError('Transport hash mismatch')
            record['value']=json.loads(record['raw'])
    return records


def batched(requests):
    """One read-only process per host; stdin avoids Linux argument-size limits."""
    groups={}
    for key,host,path in requests:
        host_key=(host.get('address'),host.get('python',sys.executable))
        group=groups.setdefault(host_key,dict(host=host,paths={}))
        group['paths'][key]=path
    records={}
    with ThreadPoolExecutor(max_workers=max(1,len(groups))) as pool:
        futures=[(g,pool.submit(fetch,g['host'],g['paths'])) for g in groups.values()]
        for group,future in futures:
            try:records.update(future.result())
            except Exception as exc:
                records.update({key:dict(error=str(exc),path=path,read_at=time.time()) for key,path in group['paths'].items()})
    return records


def protocol_ok(selection, declared):
    records=selection.get('records',[])
    return (declared==PROTOCOL and selection.get('domain')==DOMAIN and selection.get('aggregation')==AGGREGATION
            and len(records)==6 and {(r.get('view'),r.get('actor')) for r in records}=={(v,a) for v in ['DC','HC','KA'] for a in ['black','grey']}
            and all(r.get('domain')==DOMAIN for r in records))


def quality(selection):
    out={k:float(selection['metrics'][k]) for k in ['psnr','ssim','lpips']}
    if not all(math.isfinite(v) for v in out.values()):raise ValueError('Nonfinite metric')
    return out


def choose(candidates):
    if not candidates:return None
    best=min(c['scores']['lpips'] for c in candidates)
    return min((c for c in candidates if c['scores']['lpips']<=best+.001),key=lambda c:(-c['scores']['psnr'],c['metric_step'],c['id']))


def guard(selection, baseline):
    if baseline is None:return {'status':'unknown','reason':'Verified completed control missing'}
    try:
        values={}
        for actor in ['black','grey']:
            value=float(selection['by_actor'][actor]['lpips']);ref=float(baseline['by_actor'][actor]['lpips'])
            if not all(math.isfinite(x) and x>=0 for x in [value,ref]):raise ValueError('Invalid LPIPS')
            values[actor]={'lpips':value,'control_lpips':ref,'pass':value<=ref*1.03}
        return {'status':'pass' if all(v['pass'] for v in values.values()) else 'fail','actors':values}
    except (KeyError,TypeError,ValueError) as exc:return {'status':'unknown','reason':str(exc)}


def subst(text,host,name):
    return text.replace('{host_root}',host['root']).replace('{job_root}',str(Path(host['root'])/'jobs'/name))


def selector(job):
    cmd=job.get('command',[])
    try:return json.loads(cmd[cmd.index('--task')+1]).get('select_only',False)
    except (ValueError,IndexError,TypeError):return False


def collect(registry, reader=batched):
    sources={};requests=[]
    for source in registry['sources']:
        sid=source['id'];host=source.get('host',{})
        if sid in sources:raise ValueError('Duplicate source id')
        sources[sid]=source
        for name in ['manifest','state']:
            requests.append((sid+'/'+name,host,str(Path(source['controller_root'])/(name+'.json'))))
    observations=reader(requests);jobs=[];source_rows=[];next_reads=[]
    for sid,source in sources.items():
        entries={name:observations.get(sid+'/'+name,{'error':'missing response'}) for name in ['manifest','state']}
        errors=[name+': '+r['error'] for name,r in entries.items() if r.get('error')]
        source_row={'id':sid,'protocol':source['protocol'],'available':not errors,'errors':errors,
                    'provenance':{k:{a:b for a,b in v.items() if a not in ['raw','value']} for k,v in entries.items()}}
        source_rows.append(source_row)
        if errors:continue
        plan=entries['manifest']['value'];state=entries['state']['value']
        if state.get('manifest_sha256')!=digest(canonical(plan)):
            source_row['available']=False;source_row['errors'].append('State/manifest digest mismatch');continue
        for job in plan['jobs']:
            name=job['name'];key=sid+'/'+name;item=state.get('jobs',{}).get(name)
            if item is None:
                source_row['available']=False;source_row['errors'].append('Job missing from state: '+name);continue
            host=dict(plan['hosts'].get(item.get('host'),{}))
            # "local" in a remote controller means that controller's machine,
            # not the reporting machine.
            if host and not host.get('address') and source.get('host',{}).get('address'):
                host['address']=source['host']['address']
            row={'id':key,'name':name,'campaign':sid,'status':item['status'],'host':host.get('address') or item.get('host'),
                 'gpu':item.get('gpu'),'protocol':source['protocol'],'scores':None,'metric_step':None,'progress_step':None,
                 'classification':'unfinished','state_heartbeat':item.get('heartbeat'),'oom_evidence':item.get('latest_check',{}).get('oom_evidence'),
                 '_item':item,'_job':job,'_source':source}
            jobs.append(row)
            if not host or not job.get('metrics_file'):continue
            result_path=subst(job['metrics_file'],host,name);run=Path(result_path).parent
            for kind,path in [('complete',result_path),('online',str(run/'metrics.json')),('progress',str(run/'progress.json'))]:
                next_reads.append((key+'/'+kind,host,path))
    observations.update(reader(next_reads))
    review_paths=registry.get('geometry_reviews',[])
    review_obs=reader([('review/'+str(i),r.get('host',{}),r['path']) for i,r in enumerate(review_paths)]) if review_paths else {}
    observations.update(review_obs)
    reviews=[v['value'] for v in review_obs.values() if 'value' in v]
    return sources,source_rows,jobs,observations,reviews


def assemble(registry, collected):
    sources,source_rows,jobs,observations,reviews=collected
    verified=[];repeats=[];online=[];errors=[];selection_by_id={}
    for row in jobs:
        key=row['id'];item=row.pop('_item');job=row.pop('_job');source=row.pop('_source')
        progress=observations.get(key+'/progress',{}).get('value',{})
        row['progress_step']=progress.get('step');row['progress_source']=provenance(observations.get(key+'/progress',{}))
        completion=observations.get(key+'/complete',{})
        result=completion.get('value')
        if item['status']=='complete' and result and not result.get('skipped'):
            try:
                if canonical(result)!=canonical(item.get('metrics')):raise ValueError('Controller/source completion mismatch')
                selected=result['selection']
                if not protocol_ok(selected,source['protocol']):raise ValueError('Protocol mismatch or incomplete six actor/view records')
                row.update(scores=quality(selected),metric_step=int(selected['step']),progress_step=result.get('final_step',row['progress_step']),
                           config=result.get('config',{}),completion_source=provenance(completion),classification='completed_verified',
                           metric_source='completed_selected',visual_review_complete=False)
                selection_by_id[key]=selected
                if result.get('technical_smoke') or selector(job):row['classification']='technical_or_selector_excluded'
                elif result.get('config',{}).get('seed')==42:verified.append(row)
                else:row['classification']='repeat_confirmation';repeats.append(row)
            except (KeyError,TypeError,ValueError) as exc:
                row['classification']='completed_unverified';row['verification_error']=str(exc);errors.append(key+': '+str(exc))
        elif item['status']=='complete' and not result:
            row['classification']='completed_unverified';row['verification_error']=completion.get('error','Completion file absent');errors.append(key+': '+row['verification_error'])
        elif result and result.get('skipped'):row['classification']='skipped'
        if item['status'] not in ['complete','skipped','blocked']:
            record=observations.get(key+'/online',{});history=record.get('value',[])
            if isinstance(history,list) and history:
                curve=[]
                for gate in history:
                    try:
                        if not protocol_ok(gate,source['protocol']):raise ValueError('Online protocol mismatch')
                        curve.append({'step':int(gate['step']),**quality(gate)})
                    except (KeyError,TypeError,ValueError) as exc:errors.append(key+': '+str(exc))
                if curve:
                    curve.sort(key=lambda g:g['step']);latest=curve[-1]
                    row.update(scores={k:latest[k] for k in ['psnr','ssim','lpips']},metric_step=latest['step'],metric_source='online_provisional')
                    online.append({'id':key,'curve':curve,'source':provenance(record),'eligible_for_final_ranking':False})
    baseline=next((r for r in verified if r['id']==registry['control_id']),None)
    baseline_selection=selection_by_id.get(registry['control_id']) if baseline else None
    for row in verified:
        row['actor_guard']=guard(selection_by_id[row['id']],baseline_selection)
        approved=[review for review in reviews if review.get('candidate_id')==row['id']
                  and review.get('complete_sha256')==row['completion_source']['sha256']
                  and review.get('metric_step')==row['metric_step']
                  and review.get('geometry_pass') is True and review.get('visual_review_complete') is True
                  and review.get('reviewer') and review.get('reviewed_at') and review.get('evidence')]
        row['geometry_review_status']='approved_receipt' if approved else 'pending'
        if approved:row['geometry_review']=approved[-1]
    raw=choose(verified);safe=choose([r for r in verified if r['actor_guard']['status']=='pass'])
    reviewed=choose([r for r in verified if r['actor_guard']['status']=='pass' and r['geometry_review_status']=='approved_receipt'])
    recipe=lambda row:{k:v for k,v in row.get('config',{}).items() if k not in ['output','data','seed']}
    for repeat in repeats:
        match=choose([row for row in verified if recipe(row)==recipe(repeat)])
        repeat['paired_reference']=match['id'] if match else None
        repeat['paired_deltas']={k:repeat['scores'][k]-match['scores'][k] for k in ['psnr','ssim','lpips']} if match else None
    complete=all(s['available'] for s in source_rows) and not errors
    return {'time':time.time(),'protocol':PROTOCOL,'coverage_complete':complete,'sources':source_rows,'jobs':jobs,
            'completed_ranking':sorted(verified,key=lambda r:r['scores']['lpips']),
            'raw_metric_winner_among_verified_sources':raw,'best_actor_guard_passing':safe,'reviewed_geometry_candidate':reviewed,
            'repeat_confirmation':repeats,'online_provisional':online,'errors':errors,
            'automatic_promotion':False,'plans_modified':False,'visual_review_performed_by_reporter':False,
            'selection_rule':'Completed seed42 only; min masked LPIPS within .001 then higher PSNR then earlier step; per-actor3% guard',
            'coverage_note':'Missing or unreadable sources are excluded, never replaced with cached completions; winner is conditional on reported source coverage.'}


def provenance(record):return {k:v for k,v in record.items() if k not in ['raw','value']}


def markdown(report):
    text=['# Marshawn cross-campaign report','',
          'Protocol: '+report['protocol']+'. Source coverage: '+('complete' if report['coverage_complete'] else 'INCOMPLETE')+'.',
          'No automatic promotion. Online metrics are provisional and excluded from completed ranking.','']
    if report.get('stale'):text.insert(2,'STALE REPORT: refresh failed; prior observations are historical only.')
    for title,key in [('Raw numerical winner','raw_metric_winner_among_verified_sources'),('Per-actor guard passing','best_actor_guard_passing'),('Reviewed geometry candidate','reviewed_geometry_candidate')]:
        row=report[key];text.append(title+': '+(row['id']+' @'+str(row['metric_step']) if row else 'none')+'.')
    text+=['','| Campaign/job | Status | Progress | Metric step | PSNR | SSIM | LPIPS | Evidence |',
           '|---|---|---:|---:|---:|---:|---:|---|']
    for row in report['jobs']:
        q=row['scores'] or {};fmt=lambda k:format(q[k],'.5f') if k in q else '—'
        text.append('| '+ ' | '.join([row['id'],row['status'],str(row['progress_step']),str(row['metric_step']),fmt('psnr'),fmt('ssim'),fmt('lpips'),row.get('metric_source',row['classification'])])+' |')
    text+=['','Sources:']
    for source in report['sources']:
        stamp=source['provenance'].get('state',{});text.append('- '+source['id']+': '+('read' if source['available'] else 'unavailable')+', state SHA '+stamp.get('sha256','missing')+'.')
    if report['errors']:text+=['','Verification exclusions:']+['- '+e for e in report['errors']]
    return '\n'.join(text)+'\n'


def once(root,registry):
    collected=collect(registry);report=assemble(registry,collected)
    for key,record in collected[3].items():
        if 'raw' in record:
            path=root/'sources'/(key+'.json');path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_suffix('.tmp');tmp.write_text(record['raw']);os.replace(tmp,path)
    atomic(root/'report.json',report)
    path=root/'report.md';tmp=path.with_suffix('.tmp');tmp.write_text(markdown(report));os.replace(tmp,path)
    with (root/'checks.jsonl').open('a') as stream:
        stream.write(json.dumps({'time':report['time'],'coverage_complete':report['coverage_complete'],'source_count':len(report['sources']),
                                'jobs':{r['id']:{k:r.get(k) for k in ['status','progress_step','metric_step','oom_evidence']} for r in report['jobs']},
                                'raw_winner':report['raw_metric_winner_among_verified_sources']['id'] if report['raw_metric_winner_among_verified_sources'] else None,
                                'errors':report['errors']})+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['once','launch','watch','status']);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--registry',type=Path)
    args=parser.parse_args();root=args.root;root.mkdir(parents=True,exist_ok=True)
    if args.action=='status':
        record=json.loads((root/'watcher.json').read_text()) if (root/'watcher.json').exists() else {}
        print(json.dumps({'alive':alive(record.get('process')),'watcher':record}));return
    path=root/'registry.json'
    if args.registry:
        registry=json.loads(args.registry.read_text())
        if path.exists() and canonical(json.loads(path.read_text()))!=canonical(registry):raise ValueError('Registry generation is immutable')
        if not path.exists():atomic(path,registry)
    registry=json.loads(path.read_text());interval=float(registry.get('poll_seconds',60))
    if not 1<=interval<=3600:raise ValueError('Poll interval must be at most one hour')
    if args.action=='once':once(root,registry);return
    if args.action=='launch':
        if (root/'watcher.json').exists() and alive(json.loads((root/'watcher.json').read_text()).get('process')):print('Watcher already alive');return
        with (root/'watcher.log').open('ab') as stream:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'watch','--root',str(root.resolve())],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'pid':process.pid}));return
    with (root/'watcher.lock').open('a') as held:
        try:fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        atomic(root/'watcher.json',{'process':identity(),'started':time.time(),'source_sha256':digest(Path(__file__).read_bytes()),'registry_sha256':digest(path.read_bytes())})
        while True:
            try:once(root,registry)
            except Exception as exc:
                atomic(root/'last_error.json',{'time':time.time(),'error':repr(exc)})
                # Mark old report stale instead of silently presenting cached results.
                if (root/'report.json').exists():
                    stale=json.loads((root/'report.json').read_text());stale.update(coverage_complete=False,stale=True,refresh_error=repr(exc));atomic(root/'report.json',stale)
                    (root/'report.md').write_text(markdown(stale))
            time.sleep(interval)


if __name__=='__main__':main()
