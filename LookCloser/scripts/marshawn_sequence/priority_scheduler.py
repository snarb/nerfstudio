"""Global sparse24 barrier and durable work stealing; never changes neural settings."""
import argparse
import copy
import fcntl
import json
import time
from pathlib import Path
from common import atomic,read
from campaign import Host

PRIORITY=set(range(908,1268,5))

def frame(job):return int(job['id'][5:])

def pending(queue,state):
    return [j for j in queue['jobs'] if frame(j) in PRIORITY and j['id'] not in state.get('jobs',{})]

def can_receive(name,f):
    return (name!='ws07' or f<=1111) and (name!='ws08' or f>=1112)

def relocate(job,template,root):
    f=frame(job);out=copy.deepcopy(template);out['id']=job['id']
    out['command'][out['command'].index('--frame')+1]=str(f)
    out['completion']=str(root/'frames'/f'{f:06d}'/'frame_complete.json')
    out['progress']=str(root/'frames'/f'{f:06d}'/'train/progress.json')
    out.pop('start_gate',None)
    return out

def plan(queues,states,roots):
    result=copy.deepcopy(queues);moves=[]
    for name,q in result.items():
        for j in q['jobs']:
            if frame(j) not in PRIORITY:j['start_gate']=str(roots[name]/'sparse24_complete.json')
    # Take distant work first, keeping each donor's next/prepared task when possible.
    for receiver in result:
        if pending(result[receiver],states[receiver]):continue
        idle=not states[receiver].get('active')
        candidates=[]
        for donor,q in result.items():
            if donor==receiver:continue
            tasks=pending(q,states[donor])
            if len(tasks)<(1 if idle else 2):continue
            compatible=[j for j in tasks if can_receive(receiver,frame(j))]
            if compatible:candidates.append((len(tasks),donor,compatible[-1]))
        if not candidates:continue
        _,donor,job=max(candidates,key=lambda x:(x[0],x[1]))
        template=result[receiver]['jobs'][0]
        result[donor]['jobs'].remove(job)
        result[receiver]['jobs'].append(relocate(job,template,roots[receiver]))
        moves.append(dict(frame=frame(job),donor=donor,receiver=receiver))
    for name,q in result.items():
        q['jobs'].sort(key=lambda j:0 if j['id'] in states[name]['jobs'] else 1 if frame(j) in PRIORITY else 2)
    assert sorted(frame(j) for q in result.values() for j in q['jobs'])==list(range(908,1269))
    return result,moves

def recover(root,hosts):
    p=root/'priority_queue_transaction.json'
    if p.exists():
        tx=read(p)
        if not tx.get('committed'):
            for name,change in tx['changes'].items():
                h=hosts[name];current=h.get(h.agent_root/'queue.json')
                if current not in [change['before'],change['after']]:raise ValueError('Queue changed outside pending transaction: '+name)
                if current!=change['after']:h.put(h.agent_root/'queue.json',change['after'])
            tx['committed']=True;tx['committed_at']=time.time();atomic(p,tx)
    # Also clear guards left by a crash before the transaction was recorded.
    import shlex
    for h in hosts.values():h.command('rm -f '+shlex.quote(str(h.agent_root/'SCHEDULING')))

def tick(root):
    rows=read(root/'fleet.json')['hosts'];hosts={row['name']:Host(row) for row in rows if not row.get('production_disabled')}
    recover(root,hosts)
    queues={n:h.get(h.agent_root/'queue.json') for n,h in hosts.items()}
    states={n:h.get(h.agent_root/'agent_state.json') for n,h in hosts.items()}
    if any(not queues[n] or not states[n] for n in hosts):raise ValueError('Incomplete fleet observation; no scheduling mutation')
    done=set()
    for n,s in states.items():
        for k,v in s['jobs'].items():
            if k.startswith('frame') and v['status']=='complete' and int(k[5:]) in PRIORITY:
                receipt=hosts[n].get(Path(v['job']['completion']))
                if not receipt:raise ValueError('Missing completion receipt: '+k)
                done.add(int(k[5:]))
    if done==PRIORITY:
        receipt=dict(time=time.time(),frames=sorted(done),count=72,policy='All sparse24 jobs complete with completion receipts; detailed export/visual audits remain separate')
        for h in hosts.values():
            if not h.get(h.root/'sparse24_complete.json'):h.put(h.root/'sparse24_complete.json',receipt)
        atomic(root/'priority_scheduler_status.json',dict(receipt,released=True));return True
    if any(h.get(h.root/'sparse24_complete.json') for h in hosts.values()):raise ValueError('Premature global gate exists')
    new,moves=plan(queues,states,{n:h.root for n,h in hosts.items()})
    changes={n:dict(before=queues[n],after=new[n]) for n in hosts if queues[n]!=new[n]}
    if changes:
        import shlex
        for n in changes:hosts[n].put(hosts[n].agent_root/'SCHEDULING',dict(time=time.time(),owner='priority_scheduler'))
        # Let an agent already past its guard check finish publishing its start.
        time.sleep(6)
        # Revalidate before recording the transaction. Active job identities must persist.
        try:
            for n in changes:
                h=hosts[n];now=h.get(h.agent_root/'agent_state.json')
                if h.get(h.agent_root/'queue.json')!=queues[n] or now.get('jobs')!=states[n].get('jobs'):raise ValueError('Concurrent queue/worker transition; retry next tick')
        except Exception:
            for n in changes:hosts[n].command('rm -f '+shlex.quote(str(hosts[n].agent_root/'SCHEDULING')))
            raise
        tx=dict(time=time.time(),changes=changes,moves=moves,committed=False)
        atomic(root/'priority_queue_transaction.json',tx);recover(root,hosts)
        with (root/'priority_queue_history.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),moves=moves,queues_changed=sorted(changes)))+'\n')
    status=dict(time=time.time(),released=False,completed=len(done),total=72,moves=moves,pending={n:len(pending(new[n],states[n])) for n in hosts})
    atomic(root/'priority_scheduler_status.json',status);return False

def run(root,once=False):
    root=Path(root);lock=(root/'priority_scheduler.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:
            if tick(root) or once:return
        except Exception as error:
            with (root/'priority_scheduler_errors.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),error=repr(error)))+'\n')
            if once:raise
        time.sleep(30)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--once',action='store_true');a=p.parse_args();run(a.root,a.once)
