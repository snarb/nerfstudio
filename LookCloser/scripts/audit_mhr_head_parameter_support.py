"""Recompute model-only parameter-support evidence and bind the finite review."""
from pathlib import Path
import argparse
import numpy as np
from build_train_hair_semantics import read,sha,write
from probe_mhr_head_parameter_support import ROOT,MODEL


def verify():
    request=read(ROOT/'request.json');result=read(ROOT/'result.json')
    assert request['script_sha256']==sha(Path(__file__).with_name('probe_mhr_head_parameter_support.py'))
    assert request['model_sha256']==sha(MODEL/'mhr_model.pt')
    assert request['model_request_sha256']==sha(MODEL/'request.json')
    assert request['dataset_used'] is False and result['actor_fit_or_repair'] is False
    assert result['evidence_sha256']==sha(ROOT/'evidence.npz')
    assert result['panel_sha256']==sha(ROOT/'lower_jaw_parameter_variants.png')
    with np.load(ROOT/'evidence.npz') as evidence:
        base,plus,minus=[evidence[name] for name in ['base','positive','negative']]
        assert base.shape==(18439,3) and plus.shape==minus.shape==(20,18439,3)
        assert all(np.isfinite(x).all() for x in [base,plus,minus])
        derivative=(plus-minus)/2;residual=(plus+minus)/2-base
        np.testing.assert_array_equal(derivative,evidence['derivative'])
        np.testing.assert_array_equal(residual,evidence['symmetric_residual'])
        masks=dict(head=base[:,1]>=145,neck=(base[:,1]>=135)&(base[:,1]<145),
            lower_front=(base[:,1]>=145)&(base[:,1]<=153)&(base[:,2]>=0),below_neck=base[:,1]<135)
        for name,mask in masks.items():
            np.testing.assert_array_equal(mask,evidence[name]);record=result['records'][name]
            assert record['vertices']==int(mask.sum())
            rms=np.sqrt(np.mean(np.sum(derivative[:,mask]**2,axis=-1),axis=1))
            np.testing.assert_array_equal(rms,record['coefficient_rms_cm'])
            matrix=derivative[:,mask].reshape(20,-1)
            spectrum=np.sqrt(np.maximum(np.linalg.eigvalsh(matrix@matrix.T/int(mask.sum()))[::-1],0))
            np.testing.assert_allclose(spectrum,record['singular_values_cm_per_sqrt_vertex'],rtol=1e-5,atol=1e-8)
            assert float(np.linalg.norm(residual[:,mask],axis=-1).max())==record['maximum_symmetric_residual_cm']
        assert not np.any(derivative[:,masks['below_neck']])
        assert not np.any(residual[:,masks['below_neck']])


def main(action):
    verify()
    if action=='seal':
        write(ROOT/'visual_review.json',dict(reviewer='main LLM',
            inspected={str(ROOT/'lower_jaw_parameter_variants.png'):sha(ROOT/'lower_jaw_parameter_variants.png')},
            verdict='model_capacity_diagnostic_only',
            findings='Three dominant lower-front coefficients visibly vary chin/lower-face proportions. Broad neck/shoulder nearly fixed. This is not fitted DEC5 anatomy or a repair.',
            dataset_used=False,production_promoted=False))
        bindings={str(p):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
        for name in [Path(__file__).name,'probe_mhr_head_parameter_support.py']:
            p=Path(__file__).with_name(name).resolve();bindings[str(p)]=sha(p)
        for relative in ['tests/test_mhr_parameter_support.py','experiments/dec5_mhr_head_parameter_support.md']:
            p=Path(__file__).resolve().parents[1]/relative;bindings[str(p)]=sha(p)
        write(ROOT/'artifact_manifest.json',dict(bindings=bindings,status='model_only_probe_complete'))
        print('sealed',len(bindings),'bindings',flush=True)
    else:
        for path,digest in read(ROOT/'artifact_manifest.json')['bindings'].items():assert sha(path)==digest,path
        print('model support audit passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['seal','check'])
    main(parser.parse_args().action)
