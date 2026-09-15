"""Seal/recheck the finite native-witness and two-frame hard-source study."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from joint_temporal_texture import read, sha, atomic_json
from study_interior_texture_sources import ROOT, FRAMES, BASE
from diagnose_cinematic_hair_rim import OUT as WITNESSES


def verify():
    import render_smooth_temporal_mesh_video as engine
    request=engine.verify_request(ROOT)
    assert request['geometry_changed'] is False and request['production_promoted'] is False
    assert sha(BASE/'request.json')==request['matched_parent_sha256']
    for frame in FRAMES:
        for root in [BASE,ROOT]:
            folder=root/'frames'/frame;receipt=read(folder/'complete.json')
            assert receipt['request_sha256']==sha(root/'request.json')
            for name,digest in receipt['hashes'].items():assert sha(folder/name)==digest
        np.testing.assert_array_equal(np.load(BASE/'frames'/frame/'target_depth.npz')['depth'],np.load(ROOT/'frames'/frame/'target_depth.npz')['depth'])
        old=np.asarray(Image.open(BASE/'frames'/frame/'source_ids.png'));new=np.asarray(Image.open(ROOT/'frames'/frame/'source_ids.png'))
        assert np.array_equal(old==255,new==255)
    witness=WITNESSES/'001083';r=read(witness/'result.json')
    assert sha(Path(__file__).with_name('diagnose_cinematic_hair_rim.py'))==r['script_sha256']
    assert sha(witness/'projections.npz')==r['projection_sha256']
    for p,h in r['source_rgb_hashes'].items():assert sha(p)==h
    for sample in r['samples']:
        if sample['surface']:assert sha(witness/sample['panel'])==sample['panel_sha256']
    visibility=read(witness/'visibility/result.json')
    assert visibility['parent_result_sha256']==sha(witness/'result.json')
    assert sha(Path(__file__).with_name('diagnose_cinematic_hair_visibility.py'))==visibility['script_sha256']
    assert sha(witness/'visibility/evidence.npz')==visibility['evidence_sha256']
    for p,h in visibility['source_rgb_hashes'].items():assert sha(p)==h
    for sample in visibility['samples']:assert sha(witness/'visibility'/sample['panel'])==sample['panel_sha256']


def seal():
    verify()
    images=sorted((WITNESSES/'001083').glob('point_*.png'))+sorted((WITNESSES/'001083/visibility').glob('point_*.png'))+sorted((ROOT/'review').glob('*.png'))
    assert len(images)==18
    atomic_json(ROOT/'visual_review.json',dict(reviewer='main LLM',status='partial_improvement_not_production_approval',
        inspected_images={str(p):sha(p) for p in images},
        native_witness_findings='Chosen samples contain sparse-hair/background mixtures despite foreground-mask admission; alternative interior observations pass unchanged geometric visibility.',
        frames={
            '001083':dict(hair='Large reduction of tan/brown crown and side-hair rim. Jagged mesh fringe, fine stretched texture, and crown opening remain.',face='No conspicuous central-face degradation in the inspected crop.',body='Outline remains jagged; darker edge band on right neck/shoulder is more conspicuous.'),
            '001123':dict(hair='Tan outer-hair band reduced substantially. Rough polygon outline and some stretched texture remain.',face='Existing fine nose/face line remains; no hole repair is claimed.',body='Darker contour transition on right neck/shoulder; not accepted as globally artifact-free.')},
        geometry_changed=False,temporal_quality_tested=False,production_promoted=False,
        review_counts_are_not_face_metrics=True))
    bindings={}
    for root in [WITNESSES,ROOT]:
        for p in root.rglob('*'):
            if p.is_file() and p.name!='artifact_manifest.json':bindings[str(p)]=sha(p)
    for name in ['diagnose_cinematic_hair_rim.py','diagnose_cinematic_hair_visibility.py','study_interior_texture_sources.py',Path(__file__).name]:
        p=Path(__file__).with_name(name).resolve();bindings[str(p)]=sha(p)
    for relative in ['tests/test_interior_texture_sources.py','experiments/dec5_interior_texture_sources.md']:
        p=Path(__file__).resolve().parents[1]/relative;bindings[str(p)]=sha(p)
    atomic_json(ROOT/'artifact_manifest.json',dict(status='completed_two_frame_experiment_not_full_goal',bindings=bindings))
    print('sealed',len(bindings),'bindings',flush=True)


def check():
    verify();manifest=read(ROOT/'artifact_manifest.json')
    for p,h in manifest['bindings'].items():assert sha(p)==h,p
    print('checked',len(manifest['bindings']),'bindings',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['seal','check']);a=p.parse_args()
    seal() if a.action=='seal' else check()
