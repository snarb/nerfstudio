#!/usr/bin/env python3
"""Tighten existing train-only depth confidence without changing RGB or depth."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_luster_opacity import sha, write


def filter_confidence(manifest_path, output, minimum_witnesses):
    source = json.loads(manifest_path.read_text())
    if not source.get('train_only') or source.get('confidence_policy_version') != 2:
        raise ValueError('Require qualified train-only policy v2')
    if not source['minimum_witnesses'] <= minimum_witnesses <= source['source_count']:
        raise ValueError('Witness threshold must tighten the existing policy')
    output.mkdir(parents=True, exist_ok=False)
    result = dict(source)
    result.update(minimum_witnesses=minimum_witnesses, maps={},
                  parent_manifest=str(manifest_path.resolve()), parent_sha256=sha(manifest_path),
                  filter_sha256=sha(Path(__file__)),
                  confidence_filter='Existing valid AND witness count >= threshold; RGB and measured geometry unchanged')
    before, after = 0, 0
    for name, receipt in source['maps'].items():
        path = manifest_path.parent / receipt['path']
        if sha(path) != receipt['sha256']:
            raise ValueError(f'Depth receipt mismatch: {name}')
        with np.load(path) as archive:
            values = {key: archive[key].copy() for key in archive.files}
        valid, witnesses = values['valid'], values['witnesses']
        if valid.dtype != np.bool_ or witnesses.shape != valid.shape:
            raise ValueError(f'Invalid confidence array: {name}')
        kept = valid & (witnesses >= minimum_witnesses)
        values['valid'] = kept
        destination = output / path.name
        np.savez_compressed(destination, **values)
        result['maps'][name] = dict(receipt, path=destination.name, sha256=sha(destination),
                                   parent_map_sha256=receipt['sha256'],
                                   valid_pixels=int(kept.sum()), valid_fraction=float(kept.mean()))
        before += int(valid.sum())
        after += int(kept.sum())
    result['filter_summary'] = dict(maps=len(result['maps']), original_valid_pixels=before,
                                   retained_valid_pixels=after, retained_fraction=after / max(before, 1))
    write(output / 'manifest.json', result)
    print(json.dumps(result['filter_summary']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum-witnesses', type=int, default=6)
    args = parser.parse_args()
    filter_confidence(args.manifest, args.output, args.minimum_witnesses)
