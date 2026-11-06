"""Fail-closed release guard for the separate TRAIN admission candidate."""
import argparse,json,os,sys
from pathlib import Path
from marshawn_semantic_admission import sha


def verify(spec_path):
    spec_path=Path(spec_path);spec=json.loads(spec_path.read_text());gate=spec['semantic_admission'];root=Path(gate['release_receipt']).parent
    release=json.loads(Path(gate['release_receipt']).read_text())
    if (release.get('approved_for_training')is not True or release.get('spec_sha256')!=sha(spec_path)
        or release.get('semantic_manifest_sha256')!=gate['manifest_sha256']
        or any(release.get(k)is not True for k in ('native_gpu_paths_passed','all_ones_identity_passed','adam_resume_passed','source_positive_support_passed'))):raise ValueError('Semantic candidate awaits exact source/probe/technical root review')
    for name,digest in release['artifacts_sha256'].items():
        path=Path(name)
        if not path.is_absolute():path=root/path
        if sha(path)!=digest:raise ValueError('Released artifact changed: '+name)
    inventory=json.loads((root/'source_manifest.json').read_text())
    for name,digest in inventory['files_sha256'].items():
        if sha(root/'code'/name)!=digest:raise ValueError('Candidate frozen source changed: '+name)
    if gate['identity_control']or 'technical_pause_after'in spec:raise ValueError('Technical controls cannot enter production')
    return release

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--verify-only',action='store_true');a=p.parse_args();verify(a.spec)
    if not a.verify_only:os.execv(sys.executable,[sys.executable,str(Path(__file__).with_name('train_marshawn_semantic_admission.py')),'--spec',str(a.spec)])
