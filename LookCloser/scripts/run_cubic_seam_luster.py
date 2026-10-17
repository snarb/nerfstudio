"""Compose previously isolated geometric filter support with cubic sampling."""
import argparse,subprocess,sys,time
from pathlib import Path
from luster_hybrid_common import read,write
from fixed_sampling_seam_luster_hybrid import combine
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wait-pid',type=int,required=True);a=p.parse_args();root=a.root
deadline=read(root/'preflight/manifest.json')['deadline'];names=[]
for adaptive in [False,True]:
    if time.time()>deadline:raise TimeoutError('Campaign deadline')
    name='single_cubic75_seam_'+('adaptive' if adaptive else 'support')
    combine(root,'single_cubic75_fixed',name,dict(filter_support='geometry',seam_fade=3.,adaptive_seam=adaptive,seam_tolerance=.02));names.append(name)
while Path(f'/proc/{a.wait_pid}').exists():
    if time.time()>deadline:raise TimeoutError('Campaign deadline')
    time.sleep(3)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU in use'
from screen_luster_hybrid import summarize
results=[]
for name in names:
    with (root/f'logs/{name}_evaluation.log').open('w') as stream:
        subprocess.run([sys.executable,str(root/'code/LookCloser/scripts/evaluate_luster_hybrid.py'),'--root',str(root),'--candidate',name,'--device','cuda'],stdout=stream,stderr=subprocess.STDOUT,check=True)
    result=summarize(root,name);results.append(result);write(root/'evaluation/cubic75_seam_screen.json',results);print(result,flush=True)
