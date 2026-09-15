"""Fetch only the two official research checkpoint files, not a dataset.

Upstream research-only license: https://github.com/NVlabs/FoundationStereo/blob/master/LICENSE
No permission for commercial deployment is inferred by this experiment.
"""
from pathlib import Path
import subprocess
import gdown
from joint_temporal_texture import sha,atomic_json

ROOT=Path('/home/brans/lookcloser_temp/FoundationStereo/pretrained_models/23-51-11')
FILES={'cfg.yaml':'1tidGICH1_kTUUqi42aboKscuMY4IK_Xr',
       'model_best_bp2.pth':'1Yh_2o9QCUrVqZrnAXZ7RUr0zTp3JrMKe'}


if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True);records=[]
    for name,identifier in FILES.items():
        path=ROOT/name
        if not path.exists():
            print('downloading official',name,flush=True)
            result=gdown.download(id=identifier,output=str(path),quiet=True,resume=True)
            if result is None:raise RuntimeError('Official checkpoint download failed')
        records.append(dict(path=str(path),google_drive_id=identifier,bytes=path.stat().st_size,sha256=sha(path)))
        print('verified download',name,path.stat().st_size,flush=True)
    repo=ROOT.parents[1]
    atomic_json(ROOT/'download_receipt.json',dict(files=records,repository_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        repository_url='https://github.com/NVlabs/FoundationStereo',research_only=True,license_sha256=sha(repo/'LICENSE'),script_sha256=sha(__file__)))
