#!/usr/bin/env python3
"""Frozen two-expert diagnostic: expected depth only in train-voted hair.

No target image, target semantic mask, alpha cutout, or new fitting is used.
The default q10 expert is preserved exactly wherever confidence is insufficient.
"""
from pathlib import Path
import argparse
import cv2
import numpy as np
from run_dec5_sequence import read,write
from dec5_sequence_worker import sha
from study_dec5_hair_geometry import display


def gate(base,candidate):
    result=((base['hair_probability']>=.8)&(candidate['hair_probability']>=.8)
        &(base['skin_probability']<.2)&(candidate['skin_probability']<.2)
        &(candidate['support']>=np.maximum(3,base['support']))
        &(base['source_id']>=0)&(candidate['source_id']>=0)&base['roi'].astype(bool))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['baseline','candidate','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();parent=read(a.baseline/'complete.json');candidate=read(a.candidate/'complete.json')
    for key in ['checkpoint_sha256','source_rgb','source_rgb_dataset_sha256','fallback_receipt_sha256','selected_cameras']:
        if parent[key]!=candidate[key]:raise ValueError('Expert provenance differs: '+key)
    if any(v.get(k) for v in [parent,candidate] for k in ['uses_validation_rgb','uses_test_rgb']):
        raise ValueError('Held-out inputs prohibited')
    a.output.mkdir(parents=True,exist_ok=False);records=[]
    by={r['camera']:r for r in candidate['records']}
    for record in parent['records']:
        name=record['camera'];bp=a.baseline/(name+'.npz');cp=a.candidate/(name+'.npz')
        if sha(bp)!=record['npz_sha256'] or sha(cp)!=by[name]['npz_sha256']:
            raise ValueError('Expert artifact changed')
        with np.load(bp) as v:base={k:v[k] for k in v.files}
        with np.load(cp) as v:other={k:v[k] for k in v.files}
        mask=gate(base,other);result={}
        for key,value in base.items():
            changed=value.copy()
            if value.shape[:2]==mask.shape:
                if other[key].shape!=value.shape:raise ValueError('Expert shape differs')
                changed[mask]=other[key][mask]
                np.testing.assert_array_equal(changed[~mask],value[~mask])
            result[key]=changed
        result['selected_expected_depth']=mask
        output=a.output/(name+'.npz');np.savez_compressed(output,**result)
        png=a.output/(name+'.png');cv2.imwrite(str(png),np.rint(display(result['rgb'])*255).astype(np.uint8)[...,::-1])
        records.append(dict(record,npz_sha256=sha(output),png_sha256=sha(png),
            changed_domain_pixels=int(mask.sum()),gt_supported_pixels=int((result['source_id']>=0).sum())))
    write(a.output/'complete.json',dict(parent,records=records,
        method='q10/expected frozen geometry experts; train-consensus hair gate',
        depth_key='q1000_or_expected_train_consensus',
        baseline_receipt_sha256=sha(a.baseline/'complete.json'),
        expected_receipt_sha256=sha(a.candidate/'complete.json'),
        confidence_gate='both hair>=.8, skin<.2, candidate support>=max(3,parent support), both have source',
        outside_gate_preserved_exactly=True,production_promoted=False,
        target_rgb_or_semantics_loaded=False))


if __name__=='__main__':main()
