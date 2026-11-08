"""Versioned entry: original factory with explicit G1 target/role runtime."""
from pathlib import Path
import sys,argparse,fcntl,traceback,time
G=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108');sys.path.insert(0,str(G/'operations'))
from marshawn_g1_role_runtime_v138 import configure
from prepare_marshawn_tuning import write
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--frame',type=int,required=True);p.add_argument('--previous',type=int,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args();lock=open(G/'operations/source_factory_v74.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:configure().finish(a.frame,a.previous,a.prepare_only)
 except Exception as e:write(G/'operations/source_factory_v74_error.json',dict(frame=a.frame,error=str(e),traceback=traceback.format_exc(),created=time.time()));raise
