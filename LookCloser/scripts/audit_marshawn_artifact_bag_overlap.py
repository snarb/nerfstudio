"""Read-only overlap of projected artifact samples with the reviewed RGB repair.

Artifact positions are diagnostics, not new labels. Only the pre-existing 6522
qualified false-grey pixels define what the old repair would actually change.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from marshawn_bag_loss_correction import validate_manifest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(a):
    source = json.loads(a.coverage.read_text())
    manifest = validate_manifest(a.manifest, a.manifest_sha256)
    masks = {}
    counts = {}
    for row in manifest['records']:
        camera = row['camera']
        semantic = a.data / 'semantics' / (camera + '.npz')
        if sha(semantic) != row['original_semantic_sha256']:
            raise ValueError('Original semantic source drift: ' + camera)
        with np.load(semantic, allow_pickle=False) as z:
            grey, black, weight = z['grey'], z['black'], z['W64']
        exclusion = np.load(a.manifest.parent / row['exclusion'], allow_pickle=False)
        unknown = np.load(a.manifest.parent / row['unknown'], allow_pickle=False)
        chosen = exclusion & grey
        if (chosen & black).any() or (exclusion & unknown).any():
            raise ValueError('Qualified correction intersects protected source')
        if int(chosen.sum()) != row['false_grey_pixels']:
            raise ValueError('False-grey membership drift')
        masks[camera] = (chosen, exclusion, unknown, weight, grey, black)
        counts[camera] = int(chosen.sum())
    if sum(counts.values()) != 6522:
        raise ValueError('Exact reviewed correction required')
    output = []
    for point in source['points']:
        observations = []
        for projection in point['train_projections']:
            camera = projection['camera']
            if camera not in masks or not projection['in_frame']:
                continue
            x, y = projection['pixel']
            chosen, exclusion, unknown, weight, grey, black = masks[camera]
            original = float(weight[y, x])
            if abs(original - projection['W64']) > 1e-7:
                raise ValueError('Projection/main-weight correspondence drift')
            observations.append(dict(camera=camera, pixel_xy=[x, y],
                original_W64=original, corrected_W64=0. if chosen[y, x] else original,
                reviewed_false_grey=bool(chosen[y, x]),
                qualified_interior=bool(exclusion[y, x]),
                protected_unknown_shell=bool(unknown[y, x]),
                main_grey=bool(grey[y, x]), main_black=bool(black[y, x]),
                corrected_support_UNKNOWN=projection['UNKNOWN']))
        output.append(dict(label=point['label'], xyz=point['xyz'],
            changed_observations=sum(r['reviewed_false_grey'] for r in observations),
            observations=observations))
    result = dict(kind='EXISTING_6522_RGB_REPAIR_ARTIFACT_OVERLAP',
        coverage_sha256=sha(a.coverage), source_checkpoint_sha256=source['checkpoint_sha256'],
        bag_manifest_sha256=a.manifest_sha256, helper_sha256=sha(__file__),
        exact_reviewed_counts=counts, points=output, new_labels_created=False,
        masks_or_weights_modified=False, GPU_used=False,
        caveat='Projection of an old-quarter artifact is not a new bag annotation or a prediction of the current quarter+.1 rerender.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({r['label']: r['changed_observations'] for r in output}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('coverage', 'manifest', 'data', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    main(p.parse_args())
