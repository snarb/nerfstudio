import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import patch_eligibility_rank_luster_hybrid as rank
from luster_hybrid_common import read,write


def test_zero_quality_ablation_never_constructs_quality_caches(tmp_path,monkeypatch):
    base=np.full((12,12,3),.5,np.float32)
    monkeypatch.setattr(rank,'native_rows',lambda:{})
    monkeypatch.setattr(rank,'geometry',lambda *a:dict(rgb=base,alpha=np.ones((12,12)),
                                                    depth_z_q50=np.ones((12,12))))
    monkeypatch.setattr(rank,'assert_source',lambda *a:None)
    source=tmp_path/'train_photo';source.write_bytes(b'provenance')
    monkeypatch.setattr(rank,'source_path',lambda *a:source)
    def forbidden(*args):raise AssertionError('Disabled quality cache was requested')
    for name in ['quality','projected_confidence','depth_risk']:
        monkeypatch.setattr(rank,name,forbidden)
    (tmp_path/'logs').mkdir()
    write(tmp_path/'preflight/sources.json',{'sources':{'7':[8]}})
    arrays=dict(rgb=base,valid=np.ones((12,12),np.uint8),alpha=np.ones((12,12)),
                depth_error=np.zeros((12,12)),roundtrip=np.zeros((12,12)),
                uncertainty=np.zeros((12,12)),score=np.zeros((12,12)),
                resolution_ratio=np.ones((12,12)))
    for prefix in ['', 'variants/cubic75/']:
        directory=tmp_path/f'{prefix}warps/cam007/src008';directory.mkdir(parents=True)
        for key,value in arrays.items():np.save(directory/f'{key}.npy',value)
        write(directory/'receipt.json',dict(box=[0,0,12,12],arrays=list(arrays)))
    config={**rank.DEFAULT,'count':1,'source_phase_weight':0,'view_phase_weight':0,
            'source_contradiction_weight':0,'photometric_depth_weight':0}
    rank.combine(tmp_path,7,'control',config)
    result=np.load(tmp_path/'candidates/control/cam007.npy')
    assert np.array_equal(result,base)
    receipt=read(tmp_path/'candidates/control/cam007_receipt.json')
    assert len(receipt['additional_input_hashes'])==1
    assert not receipt['uses_target_rgb']
