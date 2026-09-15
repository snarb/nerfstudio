"""Pinned public mirrors after the official Drive download hit its quota.

Three mirrors publish identical checkpoint SHA-256. This is mirror agreement,
not a cryptographic attestation by the original authors. Research-only use.
"""
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
from joint_temporal_texture import sha,atomic_json

ROOT=Path('/mnt/data/dec5_foundation_model_mirror')
EXPECTED='60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1'
MIRRORS=[('yizhouzhao-nv/FoundationStereo-Backup','e4d7e21f923b12bb9a5c762bac33c6712936134c','23-51-11/model_best_bp2.pth'),
         ('pablovela5620/foundation-stereo','560e90779b20f39db3b676066def08001040ebfc','model_best_bp2.pth'),
         ('Felix-Zhenghao/FoundationStereo','204d16d379a6165e01705b5f056a7688789edfd3','model_best_bp2.pth')]


if __name__=='__main__':
    api=HfApi();records=[];ROOT.mkdir(exist_ok=True)
    for repo,revision,name in MIRRORS:
        info=api.model_info(repo,revision=revision,files_metadata=True);file=next(s for s in info.siblings if s.rfilename==name)
        assert file.lfs.sha256==EXPECTED and file.size==3298527334
        records.append(dict(repository=repo,revision=revision,file=name,sha256=file.lfs.sha256,bytes=file.size))
    repo,revision,_=MIRRORS[0];files=[]
    for name in ['23-51-11/cfg.yaml','23-51-11/model_best_bp2.pth']:
        print('downloading pinned mirror',name,flush=True)
        path=Path(hf_hub_download(repo,name,revision=revision,local_dir=ROOT))
        digest=sha(path)
        if name.endswith('.pth'):assert digest==EXPECTED
        files.append(dict(path=str(path),sha256=digest,bytes=path.stat().st_size))
        print('verified',name,path.stat().st_size,flush=True)
    atomic_json(ROOT/'receipt.json',dict(mirrors=records,files=files,script_sha256=sha(__file__),
        official_google_drive_download='failed_download_quota; no bypass',
        author_attestation_available=False,three_mirror_checkpoint_hashes_equal=True,research_only=True))
