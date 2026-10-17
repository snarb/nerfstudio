"""One-photo multiplicative residual ablation; sequential CUDA evaluation."""
import argparse,subprocess,sys,time
from pathlib import Path
from luster_hybrid_common import read,write
from log_residual_luster_hybrid import combine
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
deadline=read(root/'preflight/manifest.json')['deadline'];names=[]
for mode in ['detail','full']:
    if time.time()>deadline:raise TimeoutError('Campaign deadline')
    name='single_cubic75_log_'+mode;combine(root,'single_cubic75_fixed',name,mode,6.);names.append(name)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU in use'
from screen_luster_hybrid import summarize
results=[]
for name in names:
    with (root/f'logs/{name}_evaluation.log').open('w') as stream:
        subprocess.run([sys.executable,str(root/'code/LookCloser/scripts/evaluate_luster_hybrid.py'),'--root',str(root),'--candidate',name,'--device','cuda'],stdout=stream,stderr=subprocess.STDOUT,check=True)
    result=summarize(root,name);results.append(result);write(root/'evaluation/log_residual_screen.json',results);print(result,flush=True)
