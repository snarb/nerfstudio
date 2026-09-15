"""Isolated official MHR asset/runtime preflight, not a reconstruction repair.

No DEC5 RGB, geometry, calibration or production environment is modified.
Only public Apache-2.0 MHR assets are used; SAM3D checkpoints are not needed.
"""
from pathlib import Path
import argparse
import urllib.request
import zipfile
import numpy as np
from build_train_hair_semantics import read, sha, write

ROOT = Path('/mnt/data/dec5_mhr_head_prior_preflight')
COMMIT = 'd96fafa33bbf018647c70c3525e91f53e79d2a14'
ASSET_SHA = 'e4f4f205cd87c0fa106577ba1de4fc763e4eb197c924461d2ef7e6944e9d6b94'
URL = 'https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip'


def prepare():
    if ROOT.exists():
        raise ValueError('Preflight requires a fresh root; retained attempts are immutable')
    ROOT.mkdir()
    request = dict(upstream_commit=COMMIT, release='v1.0.1', asset_url=URL,
        github_release_asset_sha256=ASSET_SHA, script_sha256=sha(__file__),
        original_geometry_changed=False, dataset_used=False, learned_shape_prior=True,
        not_sam3d_image_inference=True, production_promoted=False)
    write(ROOT/'request.json', request)
    for name, relative in [('LICENSE', 'LICENSE'), ('upstream_README.md', 'README.md'),
        ('mhr_face_mask.ply', 'tools/mhr_smpl_conversion/assets/mhr_face_mask.ply')]:
        url = f'https://raw.githubusercontent.com/facebookresearch/MHR/{COMMIT}/{relative}'
        urllib.request.urlretrieve(url, ROOT/name)
    assert 'Apache License' in (ROOT/'LICENSE').read_text()
    urllib.request.urlretrieve(URL, ROOT/'assets.zip')
    assert sha(ROOT/'assets.zip') == ASSET_SHA
    with zipfile.ZipFile(ROOT/'assets.zip') as archive:
        assert archive.testzip() is None
        write(ROOT/'archive_inventory.json', [dict(name=i.filename, size=i.file_size) for i in archive.infolist()])
        # Extract one explicit member, never an unvalidated archive tree.
        with archive.open('assets/mhr_model.pt') as source, (ROOT/'mhr_model.pt').open('wb') as target:
            import shutil
            shutil.copyfileobj(source, target)
    write(ROOT/'assets.json', dict(files={name: sha(ROOT/name) for name in
        ['LICENSE', 'upstream_README.md', 'mhr_face_mask.ply', 'assets.zip', 'mhr_model.pt', 'archive_inventory.json']},
        request_sha256=sha(ROOT/'request.json')))
    print('assets verified', flush=True)


def probe():
    import torch
    import trimesh
    torch.set_num_threads(2)
    receipt = read(ROOT/'assets.json')
    assert receipt['request_sha256'] == sha(ROOT/'request.json')
    assert read(ROOT/'request.json')['script_sha256'] == sha(__file__)
    for name, digest in receipt['files'].items():
        assert sha(ROOT/name) == digest
    if (ROOT/'runtime.json').exists():
        raise ValueError('Runtime already attempted; preserve its outcome')
    try:
        model = torch.jit.load(str(ROOT/'mhr_model.pt'), map_location='cpu').eval()
        identity = torch.zeros(1, 45, requires_grad=True)
        pose = torch.zeros(1, 204)
        expression = torch.zeros(1, 72)
        vertices, skeleton = model(identity, pose, expression)
        assert vertices.ndim == 3 and vertices.shape[0] == 1 and vertices.shape[2] == 3
        assert torch.isfinite(vertices).all() and torch.isfinite(skeleton).all()
        # Genuine autograd check, not an inference-only compatibility claim.
        objective = vertices.square().mean()
        objective.backward()
        assert identity.grad is not None and torch.isfinite(identity.grad).all()
        assert (identity.grad[:, 20:40].abs() > 0).any()
        neutral = vertices.detach().numpy()[0]
        topology = trimesh.load(ROOT/'mhr_face_mask.ply', process=False)
        assert neutral.shape == topology.vertices.shape
        # Record alignment rather than assume the mask file is the neutral pose.
        difference = np.linalg.norm(neutral-np.asarray(topology.vertices), axis=1)
        np.savez_compressed(ROOT/'neutral.npz', vertices=neutral, triangles=np.asarray(topology.faces),
            skeleton=skeleton.detach().numpy(), head_shape_gradient=identity.grad.detach().numpy())
        mesh = trimesh.Trimesh(vertices=neutral, faces=topology.faces, process=False)
        mesh.export(ROOT/'neutral.ply')
        result = dict(status='runtime_available_anatomical_review_pending', torch_version=torch.__version__,
            vertex_count=len(neutral), triangle_count=len(topology.faces),
            source_topology_position_difference=dict(median=float(np.median(difference)), maximum=float(difference.max())),
            bounds=mesh.bounds.tolist(), identity_gradient_finite=True, head_shape_gradient_nonzero=True,
            original_geometry_changed=False, production_promoted=False,
            output_hashes={name:sha(ROOT/name) for name in ['neutral.npz', 'neutral.ply']})
    except Exception as error:
        write(ROOT/'runtime.json', dict(status='failed', error=repr(error), torch_version=torch.__version__))
        raise
    write(ROOT/'runtime.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'probe'])
    prepare() if parser.parse_args().action == 'prepare' else probe()
