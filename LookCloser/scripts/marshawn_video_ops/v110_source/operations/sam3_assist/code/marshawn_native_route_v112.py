"""Preserve any already-started BODY; CPU8packV2 only on wholly unstarted frames."""
from pathlib import Path
import argparse,json,sys,fcntl,subprocess,time
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');sys.path[:0]=[str(V/'operations'),str(V/'operations/sam3_assist/code'),str(V/'code/LookCloser/scripts')]
from prepare_marshawn_tuning import sha,write

def run(root,frame):
 root=Path(root);O=root/'operations/native_pack_cpu8_v2/routes';O.mkdir(exist_ok=True);lock=open(O/(str(frame)+'.lock'),'a+');fcntl.flock(lock,fcntl.LOCK_EX);choice=O/(str(frame)+'.json');code=root/'operations/sam3_assist/code'
 if choice.exists():r=json.loads(choice.read_text())
 else:
  current=root/'operations/sam3_assist/v5'/f'frame{frame:06d}';started=current.exists()and any(current.iterdir());version=7 if started or frame<=1700 else 112;cli=code/('marshawn_mask_assist_full_v'+str(version)+'.py');r=dict(frame=frame,version=version,cli=str(cli),cli_sha256=sha(cli),preexisting_BODY_inputs_preserved=started,created=time.time());write(choice,r)
 assert r['frame']==frame and sha(r['cli'])==r['cli_sha256'];fcntl.flock(lock,fcntl.LOCK_UN);subprocess.run([sys.executable,r['cli'],'--campaign',str(root),'--frame',str(frame)],check=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--frame',type=int,required=True);a=p.parse_args();run(a.campaign,a.frame)
