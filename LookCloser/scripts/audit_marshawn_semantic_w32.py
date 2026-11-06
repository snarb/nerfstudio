"""CPU-only comparison of the actual frozen W32/W64 sampling adapters."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    source = args.candidate / 'code/LookCloser/scripts/marshawn_tuning_sampling.py'
    sys.path.insert(0, str(source.parent))
    import marshawn_tuning_sampling as sampling
    import numpy as np
    import torch
    assert Path(sampling.__file__).resolve() == source.resolve()
    manifest = json.loads((args.candidate / 'source_manifest.json').read_text())
    assert sha(source) == manifest['files_sha256']['LookCloser/scripts/marshawn_tuning_sampling.py']
    tf = json.loads((args.data / 'transforms.json').read_text())
    train = set(tf['train_filenames'])
    names = [args.data / row['file_path'] for row in tf['frames'] if row['file_path'] in train]
    assert len(names) == len(set(names)) == 63

    def build(width):
        pipeline = SimpleNamespace(
            datamanager=SimpleNamespace(
                train_pixel_sampler=SimpleNamespace(sample=lambda *a, **k: None, sample_count=0),
                train_dataset=SimpleNamespace(image_filenames=names)),
            model=SimpleNamespace(config=SimpleNamespace(reconstruction_loss_type='charbonnier'),
                                  get_loss_dict=lambda *a, **k: {}))
        return sampling.SamplingAdapter(pipeline, args.data, 'actor', 'body80', width)

    narrow, wide = build(32), build(64)
    assert narrow.source_records == wide.source_records
    records = []
    for a, b, source_row in zip(narrow.rows, wide.rows, narrow.source_records):
        for key in ('core', 'band', 'valid', 'black', 'grey', 'actor_union'):
            np.testing.assert_array_equal(a[key], b[key])
        for key in ('core', 'band', 'valid'):
            np.testing.assert_array_equal(a['pools'][key], b['pools'][key])
        np.testing.assert_array_equal(a['fas']['probabilities'], b['fas']['probabilities'])
        assert len(a['fas']['cells']) == len(b['fas']['cells'])
        for x, y in zip(a['fas']['cells'], b['fas']['cells']):
            assert len(x) == len(y)
            for xx, yy in zip(x, y):
                np.testing.assert_array_equal(xx, yy)
        w32, w64 = a['weight'], b['weight']
        assert np.isfinite(w32).all() and np.isfinite(w64).all()
        assert ((w32 >= 0) & (w32 <= w64 + 1e-6) & (w64 <= 1)).all()
        lost = a['actor_union'] & (w64 > 0) & (w32 == 0)
        assert not lost.any()
        records.append(dict(camera=source_row['camera'],
                            positive32=int((w32 > 0).sum()), positive64=int((w64 > 0).sum()),
                            newly_zero_actor_pixels=int(lost.sum()),
                            proposal_means32=a['proposal_means'].tolist(),
                            proposal_means64=b['proposal_means'].tolist()))
    expectations = {}
    for step in (0, 4096, 8192, 30376):
        counts = sampling.body80_quotas(4096, step)
        values = [float(np.mean([row['proposal_means'] for row in adapter.rows], axis=0) @ counts / 4096)
                  for adapter in (narrow, wide)]
        assert 0 < values[0] <= values[1] <= 1
        expectations[str(step)] = dict(exact_integer_source_counts=counts.tolist(), W32=values[0], W64=values[1])
    assert not torch.cuda.is_initialized()
    result = dict(passed=True, CPU_only=True, actual_frozen_SamplingAdapter_exercised=True,
                  sampling_source_sha256=sha(source), script_sha256=sha(__file__),
                  data_transforms_sha256=sha(args.data / 'transforms.json'),
                  source_manifest_sha256=sha(args.candidate / 'source_manifest.json'),
                  cameras=63, same_masks_proposal_pools_FAS=True,
                  W32_monotone_subset_of_W64=True, newly_zero_actor_pixels=0,
                  positive_analytic_normalizers=expectations,
                  GPU_model_or_optimizer_created=False, records=records,
                  limitation='A smaller loss band may still reduce unsegmented hair evidence; this is a runtime/data parity proof, not a quality result.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}))


if __name__ == '__main__':
    main()
