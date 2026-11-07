"""Idempotent frame pipeline, executed inside the fleet supervisor GPU lease.

Commands are argv arrays. Each stage must validate existing outputs before
returning success; a receipt alone is never a reason to skip its validator.
"""
from pathlib import Path
import argparse, fcntl, json, os, signal, subprocess, sys, time
from marshawn_video_train import atomic, sha, read, verify_code


def run(campaign, frame, pipeline_path, pipeline_sha256):
    campaign=Path(campaign).resolve();root=campaign/'frames'/('%06d'%frame);root.mkdir(parents=True,exist_ok=True)
    held=(root/'pipeline.lock').open('a+')
    try:fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another process owns this frame pipeline')
    if sha(pipeline_path)!=pipeline_sha256:raise ValueError('Immutable pipeline command set changed')
    verify_code(campaign);pipeline=read(pipeline_path)
    if pipeline.get('schema')!=1:raise ValueError('Unknown frame pipeline schema')
    values=dict(campaign_root=str(campaign),frame_root=str(root),frame=str(frame),
                code=str(campaign/'code/LookCloser/scripts'),python=sys.executable)
    def substitute(value):
        for key,text in values.items():value=value.replace('{'+key+'}',text)
        return value
    env=os.environ.copy();env['PYTHONPATH']=str(campaign/'code')+os.pathsep+str(campaign/'code/LookCloser/scripts')
    lease_fd=os.environ.get('LOOKCLOSER_GPU_LEASE_FD')
    inherited=()
    if lease_fd is not None:
        lease_fd=int(lease_fd);os.fstat(lease_fd);inherited=(lease_fd,)
    current=[];requested_stop=[]
    def forward(signum,frame_object):
        requested_stop.append(signum)
        if current and current[0].poll() is None:current[0].send_signal(signum)
    signal.signal(signal.SIGTERM,forward);signal.signal(signal.SIGINT,forward)
    for stage in pipeline['stages']:
        name=stage['name']
        if not name.replace('_','').isalnum():raise ValueError('Unsafe stage name')
        command=[substitute(s) for s in stage['command']]
        stage_env=env.copy()
        stage_env.update({key:substitute(value) for key,value in stage.get('env',{}).items()})
        progress=dict(frame=frame,stage=name,status='running',time=time.time(),
                      pipeline_sha256=sha(pipeline_path),command=command)
        atomic(root/'pipeline_progress.json',progress)
        with (root/(name+'.log')).open('ab') as stream:
            child=subprocess.Popen(command,env=stage_env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                                   pass_fds=inherited)
            current[:]=[child];code=child.wait();current.clear()
        if code:raise SystemExit(code if code>0 else 128-code)
        if requested_stop:raise SystemExit(128+requested_stop[-1])
        if stage.get('receipt') and not Path(substitute(stage['receipt'])).is_file():
            raise RuntimeError('Stage returned without required receipt: '+name)
    done=read(root/'frame_completion.json')
    if done.get('frame')!=frame or done.get('technical') is not False:raise ValueError('Missing production completion')
    atomic(root/'pipeline_progress.json',dict(frame=frame,stage='complete',status='complete',time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--frame',type=int,required=True);p.add_argument('--pipeline',type=Path,required=True)
    p.add_argument('--pipeline-sha256',required=True)
    a=p.parse_args();run(a.campaign,a.frame,a.pipeline,a.pipeline_sha256)
