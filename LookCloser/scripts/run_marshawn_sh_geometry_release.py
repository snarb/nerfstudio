"""Hash-bound release gate for the separate SH geometry controller; no implicit approval."""
import argparse,hashlib,json,os,sys
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def verify(spec_path):
    spec_path=spec_path.resolve();root=spec_path.parent;spec=json.loads(spec_path.read_text())
    release=Path(spec['prefix_candidate']['release_receipt']);value=json.loads(release.read_text())
    required=dict(approved_for_training=True,native_amp_zero_update_passed=True,technical_resume_passed=True,correct_sh_directions=True)
    if any(value.get(k)is not v for k,v in required.items()):raise ValueError('Explicit root production release required')
    if value.get('spec_sha256')!=sha(spec_path) or value.get('band_weight')!=spec['band_weight']:raise ValueError('Released spec changed')
    source=root/'source_manifest.json'
    if value.get('source_manifest_sha256')!=sha(source):raise ValueError('Released source inventory changed')
    inventory=json.loads(source.read_text());code=(root/'code').resolve()
    for name,digest in inventory['files'].items():
        p=(code/name).resolve()
        if not p.is_relative_to(code) or sha(p)!=digest:raise ValueError('Frozen source changed: '+name)
    evidence=value.get('evidence_sha256',{})
    if not evidence or not any('verification_'in p for p in evidence) or not any('probe'in p for p in evidence):raise ValueError('Actual probe and Adam-resume receipts required')
    for name,digest in evidence.items():
        p=Path(name)
        if not p.is_absolute():p=root/p
        if sha(p)!=digest:raise ValueError('Released evidence changed: '+name)
    return root/'code/LookCloser/scripts/train_marshawn_sh_geometry.py'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);a=p.parse_args();worker=verify(a.spec)
    os.execv(sys.executable,[sys.executable,str(worker),'--spec',str(a.spec)])
