"""CPU proof: inherited GPU lease survives parent exit until its child exits."""
from pathlib import Path
import fcntl,json,os,signal,subprocess,sys,tempfile,time


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);lock=root/'lease.lock';ready=root/'ready.json'
        # The outer owner simulates the fleet supervisor. Its child simulates
        # the frame worker, which forwards the SAME FD again to a stage child.
        stage="import os,json,time,sys; from pathlib import Path; fd=int(os.environ['LOOKCLOSER_GPU_LEASE_FD']); os.fstat(fd); Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid(),'fd':fd})); time.sleep(60)"
        frame="import os,subprocess,sys; fd=int(os.environ['LOOKCLOSER_GPU_LEASE_FD']); subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]],pass_fds=(fd,),start_new_session=True)"
        owner="import fcntl,os,subprocess,sys; stream=open(sys.argv[1],'a+'); fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB); env=dict(os.environ,LOOKCLOSER_GPU_LEASE_FD=str(stream.fileno())); subprocess.run([sys.executable,'-c',sys.argv[2],sys.argv[3],sys.argv[4]],env=env,pass_fds=(stream.fileno(),),check=True)"
        process=subprocess.Popen([sys.executable,'-c',owner,str(lock),frame,stage,str(ready)])
        process.wait(timeout=10)
        deadline=time.monotonic()+10
        while not ready.exists() and time.monotonic()<deadline:time.sleep(.02)
        if not ready.exists():raise AssertionError('Stage child failed to inherit/open the lease FD')
        child=json.loads(ready.read_text());held=lock.open('a+')
        try:
            try:fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:raise AssertionError('Second optimizer could claim lease while orphan stage survives')
        finally:os.kill(child['pid'],signal.SIGTERM)
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            try:fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(.02)
        else:raise AssertionError('Lease did not release when final child exited')
        print(json.dumps(dict(status='PASS',outer_parent_exited=True,frame_parent_exited=True,
                              live_stage_prevented_second_lease=True,lease_released_after_stage_exit=True)))


if __name__=='__main__':main()
