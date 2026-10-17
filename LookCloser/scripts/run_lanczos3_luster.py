"""Sequential GPU evaluation after a fixed-label native interpolation comparison."""
import argparse,subprocess,sys,time
from pathlib import Path
from luster_hybrid_common import read,write
from lanczos3_luster_hybrid import warp
from lanczos3_fixed_luster_hybrid import combine
from fixed_sampling_seam_luster_hybrid import combine as seam
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wait-pids',nargs='+',type=int,default=[]);a=p.parse_args();root=a.root
deadline=read(root/'preflight/manifest.json')['deadline']
for target,sources in read(root/'preflight/sources.json')['sources'].items():
    for source in sources[:16]:
        if time.time()>deadline:raise TimeoutError('Campaign deadline')
        warp(root,int(target),source)
combine(root,'single_risk_patch','single_lanczos3_fixed')
name='single_lanczos3_seam'
seam(root,'single_lanczos3_fixed',name,dict(filter_support='geometry',seam_fade=3.,adaptive_seam=True,seam_tolerance=.02))
while any(Path(f'/proc/{pid}').exists() for pid in a.wait_pids):
    if time.time()>deadline:raise TimeoutError('Campaign deadline')
    time.sleep(3)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU in use'
from screen_luster_hybrid import summarize
with (root/f'logs/{name}_evaluation.log').open('w') as stream:
    subprocess.run([sys.executable,str(root/'code/LookCloser/scripts/evaluate_luster_hybrid.py'),'--root',str(root),'--candidate',name,'--device','cuda'],stdout=stream,stderr=subprocess.STDOUT,check=True)
result=summarize(root,name);write(root/'evaluation/lanczos3_screen.json',[result]);print(result,flush=True)
