"""Bilinear color with cubic footprint visibility: isolate interpolation effect."""
from pathlib import Path
import argparse
from luster_hybrid_common import read,write,sha
from risk_single_luster_hybrid import combine

def run(root):
    variant=root/'variants/cubic_visibility';variant.mkdir(exist_ok=True)
    for key in ['preflight','geometry','baseline','evaluation','candidates','logs','runtime','code']:
        if not (variant/key).exists():(variant/key).symlink_to(root/key,target_is_directory=True)
    config=read(root/'candidates/single_risk_patch/config.json')
    for target,sources in read(root/'preflight/sources.json')['sources'].items():
        for source in sources[:config['count']]:
            rel=Path(f'cam{int(target):03d}/src{source:03d}');parent=root/'variants/cubic4/warps'/rel;original=root/'warps'/rel;out=variant/'warps'/rel;r=read(parent/'receipt.json')
            if (out/'receipt.json').exists():
                assert read(out/'receipt.json')['control_code_sha256']==sha(Path(__file__));continue
            out.mkdir(parents=True)
            for key in r['arrays']:(out/f'{key}.npy').symlink_to((original if key=='rgb' else parent)/f'{key}.npy')
            write(out/'receipt.json',{**r,'resampling':'native_bilinear_cubic_visibility_control','control_code_sha256':sha(Path(__file__)),'cubic_parent_receipt_sha256':sha(parent/'receipt.json'),'array_hashes':{k:sha(out/f'{k}.npy') for k in r['arrays']}})
        combine(variant,int(target),'single_cubic_visibility_control',config)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
