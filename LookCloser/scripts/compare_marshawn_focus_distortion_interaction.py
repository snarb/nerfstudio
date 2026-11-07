"""CPU-only four-corner comparison reusing already sealed 15188 ray audits."""
import argparse,hashlib,json,os,shutil,subprocess,time
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def main(policy_path):
    policy=json.loads(policy_path.read_text());output=Path(policy['output']);output.mkdir(exist_ok=True)
    for path,digest in policy['helpers_sha256'].items():
        if sha(path)!=digest:raise ValueError('Bound CPU comparison helper changed')
    pending=Path(policy['new_audit'])
    while not(pending/'complete.json').exists():
        (output/'state.json').write_text(json.dumps(dict(pid=os.getpid(),phase='waiting_new15188audit',time=time.time()))+'\n');time.sleep(30)
    records=[]
    for reference in policy['references']:
        source=Path(reference['audit']);record=next(r for r in json.loads((source/'selected_models.json').read_text())if r['name']==reference['name'])
        if record['step']!=15188:raise ValueError('Only matched15188 corners allowed')
        expected=reference.get('checkpoint_sha256')
        if expected is not None and record['files_sha256']['checkpoint.ckpt']!=expected:raise ValueError('Predeclared corner changed')
        archive=Path(record['archive'])
        for name,digest in record['files_sha256'].items():
            if sha(archive/name)!=digest:raise ValueError('Protected source changed: '+str(archive/name))
        for label in ['clothing103','DC12']:
            audit=json.loads((archive/(label+'_trace/rays/audit.json')).read_text())
            if not audit['frozen_state_unchanged']or audit['checkpoint_sha256']!=record['files_sha256']['checkpoint.ckpt']:raise ValueError('Trace identity differs')
        # Keep source arrays at their original immutable path; copy only review RGBs.
        shutil.copytree(archive/'renders',output/record['name']/'renders',dirs_exist_ok=True)
        records.append(record)
    (output/'selected_models.json').write_text(json.dumps(records,indent=2)+'\n')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    helpers=Path(policy['helpers'])
    for script,label in [('compare_selected_traces.py','clothing103'),('compare_selected_dc.py','DC12')]:
        subprocess.run([policy['python'],str(helpers/script),'--campaign',policy['campaign'],'--output',str(output/label)],env=env,check=True)
    subprocess.run([policy['python'],str(helpers/'summarize_marshawn_depth_residuals.py'),'--comparison',str(output/'clothing103/comparison.json'),'--output',str(output/'clothing103/depth_residuals.json')],env=env,check=True)
    (output/'complete.json').write_text(json.dumps(dict(passed=True,policy_sha256=sha(policy_path),GPU_queries=0,
        existing_three_corners_reused_without_replay=True,matched_step=15188,geometry_quality_approved=False),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--policy',type=Path,required=True);main(p.parse_args().policy)
