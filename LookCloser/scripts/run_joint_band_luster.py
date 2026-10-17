"""Test selective band admission after earlier GPU owners finish."""
import argparse,subprocess,sys,time
from pathlib import Path
from luster_hybrid_common import read,write
from joint_band_luster_hybrid import combine
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wait-pids',nargs='+',type=int,default=[]);a=p.parse_args();root=a.root
deadline=read(root/'preflight/manifest.json')['deadline'];names=[]
for suffix,margin,joint in [('mid25',.25,False),('joint25',.25,True),('joint10',.1,True)]:
    if time.time()>deadline:raise TimeoutError('Campaign deadline')
    name='single_joint_band_'+suffix
    combine(root,'single_source_phase_both_rt2_seam',name,margin,joint);names.append(name)
while any(Path(f'/proc/{pid}').exists() for pid in a.wait_pids):
    if time.time()>deadline:raise TimeoutError('Campaign deadline')
    time.sleep(3)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU in use'
from screen_luster_hybrid import summarize
results=[]
for name in names:
    if time.time()>deadline:raise TimeoutError('Campaign deadline')
    with (root/f'logs/{name}_evaluation.log').open('w') as stream:
        subprocess.run([sys.executable,str(root/'code/LookCloser/scripts/evaluate_luster_hybrid.py'),'--root',str(root),'--candidate',name,'--device','cuda'],stdout=stream,stderr=subprocess.STDOUT,check=True)
    result=summarize(root,name);results.append(result);write(root/'evaluation/joint_band_screen.json',results);print(result,flush=True)
