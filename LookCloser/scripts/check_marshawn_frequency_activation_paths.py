"""Validate changed host paths using the frozen fitter's pre-Torch validate-only branch."""
import argparse,contextlib,hashlib,importlib.util,io,json,sys
from pathlib import Path
from types import SimpleNamespace


def main(root,host,output):
    if output.exists():raise ValueError('Preserve prior path check')
    p=root/'science/worker.py';s=importlib.util.spec_from_file_location('frozen_frequency_worker',p);worker=importlib.util.module_from_spec(s);s.loader.exec_module(worker)
    inventory=json.loads((root/'jobs.json').read_text());checked=[]
    for job in inventory:
        if job['host']!=host:continue
        path=Path(job['spec']);assert hashlib.sha256(path.read_bytes()).hexdigest()==job['spec_sha256']
        with contextlib.redirect_stdout(io.StringIO()):worker.main(SimpleNamespace(plan=path,validate_only=True,pinned_worker=True,lease_receipt=None,gpu=None))
        spec=json.loads(path.read_text());assert not Path(spec['output']).exists()
        checked.append(job['name'])
    assert 'torch'not in sys.modules
    output.write_text(json.dumps(dict(passed=True,host=host,jobs=len(checked),checked=checked,Torch_imported=False,CUDA_initialized=False,GPU_jobs_launched=False,scope='Only newly materialized host/source/output paths. No unchanged scientific tests or GPU factory rerun.'),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--host',choices=['local','silver'],required=True);a=p.parse_args();main(a.root,a.host,a.output)
