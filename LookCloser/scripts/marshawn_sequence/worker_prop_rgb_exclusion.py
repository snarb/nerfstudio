"""Causal ablation: omit conflicting prop RGB in explicitly named cameras."""
import argparse
from pathlib import Path
import numpy as np
from common import atomic, read
import worker_prop_guard as guard


def inside_capsule(x, y, endpoints, radius):
    p, q = np.asarray(endpoints, dtype=np.float64)
    v = q-p
    t = np.clip(((x-p[0])*v[0]+(y-p[1])*v[1])/max(float(v@v), 1e-12), 0, 1)
    return (x-p[0]-t*v[0])**2+(y-p[1]-t*v[1])**2 <= radius**2


def train(spec_path):
    spec = read(spec_path)
    manifest = read(spec['prop_guard_manifest'])
    excluded = set(spec['prop_rgb_excluded_cameras'])
    if not excluded or not excluded <= set(manifest['cameras']):
        raise ValueError('Explicit existing camera names required')
    original_install = guard.background.install
    def install(pipe, parameters):
        original_install(pipe, parameters)
        names = [Path(p).stem for p in pipe.datamanager.train_dataset.image_filenames]
        original = pipe.model.get_loss_dict
        def loss(outputs, batch, metrics_dict=None):
            if pipe.model.training and 'actor_confidence' in batch:
                indices = batch['indices'].detach().cpu().numpy()
                weights = batch['actor_confidence'].clone()
                for camera in excluded:
                    ids = np.flatnonzero(indices[:, 0] == names.index(camera))
                    row = manifest['cameras'][camera]
                    selected = inside_capsule(indices[ids, 2], indices[ids, 1], row['endpoints_xy'], row['radius_px'])
                    weights[ids[selected]] = 0
                batch = dict(batch, actor_confidence=weights)
            return original(outputs, batch, metrics_dict)
        pipe.model.get_loss_dict = loss
        atomic(Path(parameters['output'])/'prop_rgb_exclusion_audit.json', dict(
            excluded_cameras=sorted(excluded), train_cameras=names,
            region='same conservative projected capsule as background guard',
            objective='Omit these RGB observations; preserve original fixed normalizer and all other terms',
            synthetic_rgb=False))
    guard.background.install = install
    guard.train(spec_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    train(parser.parse_args().spec)
