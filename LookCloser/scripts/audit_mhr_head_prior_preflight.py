"""Verify public prior runtime/assets/anatomical review; no DEC5 fit approval."""
from pathlib import Path
import argparse
import zipfile
import numpy as np
from build_train_hair_semantics import read, sha, write
from prepare_mhr_head_prior import ROOT, ASSET_SHA


def verify():
    import torch
    import trimesh
    torch.set_num_threads(2)
    request = read(ROOT/'request.json')
    assert request['script_sha256'] == sha(Path(__file__).with_name('prepare_mhr_head_prior.py'))
    assert not request['dataset_used'] and not request['original_geometry_changed']
    assert sha(ROOT/'assets.zip') == ASSET_SHA
    assets = read(ROOT/'assets.json')
    assert assets['request_sha256'] == sha(ROOT/'request.json')
    for name, digest in assets['files'].items():
        assert sha(ROOT/name) == digest
    review = read(ROOT/'review/manifest.json')
    assert review['script_sha256'] == sha(Path(__file__).with_name('review_mhr_head_prior.py'))
    assert review['runtime_sha256'] == sha(ROOT/'runtime.json')
    assert review['open_edges'] == review['nonmanifold_edges'] == 0
    for name, digest in review['files'].items():
        assert sha(name) == digest
    for name, digest in read(ROOT/'runtime.json')['output_hashes'].items():
        assert sha(ROOT/name) == digest
    model = torch.jit.load(str(ROOT/'mhr_model.pt'), map_location='cpu').eval()
    buffers = dict(model.named_buffers())
    topology = trimesh.load(ROOT/'mhr_face_mask.ply', process=False)
    np.testing.assert_array_equal(buffers['character_torch.mesh.faces'].numpy(), topology.faces)
    with torch.no_grad():
        vertices, _ = model(torch.zeros(1,45), torch.zeros(1,204), torch.zeros(1,72))
    np.testing.assert_array_equal(vertices.numpy()[0], np.load(ROOT/'neutral.npz')['vertices'])
    with zipfile.ZipFile(ROOT/'assets.zip') as archive:
        license_bytes = archive.read('assets/LICENSE.txt')
    assert b'Apache License' in license_bytes and b'Version 2.0' in license_bytes
    return license_bytes


def main(action):
    license_bytes = verify()
    if action == 'seal':
        # Preserve the actual release license, in addition to the repository one.
        with zipfile.ZipFile(ROOT/'assets.zip') as archive:
            archive.extract('assets/LICENSE.txt', ROOT/'release_license')
        write(ROOT/'visual_review.json', dict(reviewer='main LLM',
            inspected={str(ROOT/'review/anatomy.png'):sha(ROOT/'review/anatomy.png')},
            verdict='anatomical_domain_suitable_for_a_bounded_fit_test_not_repair_approval',
            findings='Neutral front, side and low oblique clay show continuous underside of jaw, neck and shoulders, unlike the open-front MediaPipe face. Coarse generic anatomy, not this actor. No obvious topology corruption in inspected panels.',
            exact_model_buffer_topology_verified=True, exact_neutral_replay=True,
            fit_to_actor=False, geometry_repaired=False, production_promoted=False))
        bindings={str(p):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
        for name in ['prepare_mhr_head_prior.py', 'review_mhr_head_prior.py', Path(__file__).name]:
            p=Path(__file__).with_name(name).resolve(); bindings[str(p)]=sha(p)
        p=Path(__file__).resolve().parents[1]/'experiments/dec5_mhr_head_prior_preflight.md'
        bindings[str(p)]=sha(p)
        write(ROOT/'artifact_manifest.json', dict(bindings=bindings, status='preflight_only_complete'))
        print('sealed',len(bindings),'bindings',flush=True)
    else:
        for path,digest in read(ROOT/'artifact_manifest.json')['bindings'].items():
            assert sha(path)==digest,path
        print('checked preflight; no DEC5 geometry changed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['seal','check'])
    main(parser.parse_args().action)
