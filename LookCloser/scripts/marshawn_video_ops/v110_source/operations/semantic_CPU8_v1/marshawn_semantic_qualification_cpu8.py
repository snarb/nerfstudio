"""Parallel execution of the qualified v105 semantic ROI function, unchanged.

The caller owns the canonical source transaction. Workers only read an immutable
camera context; the parent merges results in the original binding/ROI order.
An explicit output directory supports qualification against a private snapshot.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107')
CORE_SHA256 = '838e6b701e152d6591a1dbe7db33fd0196238ea0d0f2a80df56f2ef2cc4a5831'
_CORE = None
_DATA = None
_FRAME = None
_THREAD_LIMIT = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


class _ReadOnlyArchive:
    """Lazy NPZ arrays retained only for one camera's repeated ROI reads."""
    def __init__(self, archive):
        self.archive = archive
        self.files = list(archive.files)
        self.arrays = {}

    def __getitem__(self, key):
        if key not in self.arrays:
            array = self.archive[key]
            array.setflags(write=False)
            self.arrays[key] = array
        return self.arrays[key]

    def __contains__(self, key):
        return key in self.files

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def close(self):
        # Existing "with np.load" blocks must not invalidate later ROI reads.
        pass

    def release(self):
        self.arrays.clear()
        self.archive.close()


@contextlib.contextmanager
def camera_archives(data, frame, camera):
    allowed = set()
    for number in [frame, frame - 5]:
        for folder in ['native_masks', 'person_candidates', 'person_candidates_sg']:
            path = data / 'frames' / f'{number:06d}' / folder / (camera + '.npz')
            if path.exists():
                allowed.add(path.resolve())
    original = np.load
    archives, bindings = {}, {}

    def load(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)):
            path = Path(file).resolve()
            if path in allowed and not args and not kwargs.get('mmap_mode'):
                if path not in archives:
                    digest = sha(path)
                    archives[path] = _ReadOnlyArchive(original(file, **kwargs))
                    bindings[str(path)] = digest
                return archives[path]
        return original(file, *args, **kwargs)

    np.load = load
    try:
        yield bindings
        for path, digest in bindings.items():
            if sha(path) != digest:
                raise ValueError('Immutable camera context changed: ' + path)
    finally:
        np.load = original
        for archive in archives.values():
            archive.release()


def _initialize(source, data, frame, core_source, expected_core_sha):
    global _CORE, _DATA, _FRAME, _THREAD_LIMIT
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    cv2.setNumThreads(1)
    from threadpoolctl import threadpool_limits
    _THREAD_LIMIT = threadpool_limits(limits=1)
    source, _DATA, _FRAME = Path(source), Path(data), frame
    for path in [source / 'operations', source / 'code/LookCloser/scripts', source / 'code']:
        sys.path.insert(0, str(path))
    if sha(core_source) != expected_core_sha:
        raise ValueError('Qualified semantic function source changed')
    spec = importlib.util.spec_from_file_location('qualified_semantic_cpu_core', core_source)
    _CORE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_CORE)
    _CORE.V = _DATA


def _camera(task):
    camera, rows = task
    started = time.time()
    _CORE.CREW_CACHE.clear()
    result = []
    with camera_archives(_DATA, _FRAME, camera) as inputs:
        for index, row in rows:
            result.append((index, _CORE.qualify_tile(_FRAME, row)))
    _CORE.CREW_CACHE.clear()
    return result, dict(camera=camera, elapsed_seconds=time.time() - started,
                        input_files_sha256=inputs, pid=os.getpid())


def qualify(frame, bindings, root=ROOT, output=None, workers=8, data_root=None,
            core_source=None, expected_core_sha=CORE_SHA256):
    root, bindings = Path(root), Path(bindings)
    data = Path(data_root) if data_root else root
    core_source = Path(core_source) if core_source else root / 'operations/marshawn_current_parts_v105.py'
    if not 1 <= workers <= 8:
        raise ValueError('CPU workers must be between1 and8')
    if sha(core_source) != expected_core_sha:
        raise ValueError('Qualified semantic function source changed')
    if output is None and (data / 'frames' / f'{frame:06d}' / 'native_semantic_ready.json').exists():
        raise ValueError('Published source requires an explicit private output')
    output = Path(output) if output else root / 'operations/current_native_parts_v105' / f'frame{frame:06d}'
    output.mkdir(parents=True, exist_ok=True)
    digest = sha(bindings)
    source = json.loads(bindings.read_text())
    if source['frame'] != frame:
        raise ValueError('Bindings contain a different frame')
    grouped, inputs = {}, {}
    for index, row in enumerate(source['records']):
        camera = row['camera']
        grouped.setdefault(camera, []).append((index, row))
        native = data / 'frames' / f'{frame:06d}' / 'native_masks' / (camera + '.npz')
        inputs.setdefault(camera, sha(native))
    started = time.time()
    context = __import__('multiprocessing').get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=context, initializer=_initialize,
        initargs=(str(root), str(data), frame, str(core_source), expected_core_sha),
    ) as pool:
        completed = list(pool.map(_camera, grouped.items()))
    by_index, timings = {}, []
    for values, timing in completed:
        timings.append(timing)
        for index, value in values:
            if index in by_index:
                raise ValueError('Duplicate ROI result')
            by_index[index] = value
    rows, masks = [], {}
    for index, row in enumerate(source['records']):
        camera = row['camera']
        x0, y0, x1, y1 = row['ROI']
        for item in by_index[index]:
            masks.setdefault(camera, np.zeros((4320, 8192), bool))[y0:y1, x0:x1] |= item['mask']
            rows.append(dict(camera=camera, ROI=row['ROI'],
                             NN_pair={k: v for k, v in item.items() if k != 'mask'},
                             spec=row['spec'], spec_sha256=sha(row['spec']),
                             candidates=row['candidates'],
                             NN_receipt_sha256=sha(Path(row['candidates']) / 'receipt.json'),
                             proposal=row['proposal'], proposal_sha256=row['proposal_sha256']))
    for camera, mask in masks.items():
        native = data / 'frames' / f'{frame:06d}' / 'native_masks' / (camera + '.npz')
        if sha(native) != inputs[camera]:
            raise ValueError('Preapplication native changed: ' + camera)
        with np.load(native) as archive:
            mask &= ~archive['qualified_actor_rgb'] if 'qualified_actor_rgb' in archive.files else True
        path = output / (camera + '_qualified_new_union.npz')
        with path.with_suffix('.npz.tmp').open('wb') as stream:
            np.savez_compressed(stream, qualified_new_union=mask)
        os.replace(path.with_suffix('.npz.tmp'), path)
    if sha(bindings) != digest:
        raise ValueError('Bindings changed during qualification')
    result = dict(kind='CURRENT_NATIVE_SEMANTIC_PAIR_UNION_QUALIFICATION_V105', frame=frame,
                  records=rows, input_native_sha256=inputs,
                  qualified_archives={camera: dict(path=str(output / (camera + '_qualified_new_union.npz')),
                     sha256=sha(output / (camera + '_qualified_new_union.npz')),
                     array_sha256=hashlib.sha256(mask.tobytes()).hexdigest(), pixels=int(mask.sum()))
                     for camera, mask in masks.items()},
                  bindings=str(bindings), bindings_sha256=digest,
                  scientific_constraints=dict(scores_min=.7, different_prompt_pair_IoU_min=.85,
                     prior_union_identity_and_ROI_only=True, current_NN_component64_bound=True,
                     known_current_CREW_core_excluded=True, union_ownership_not_claimed=True),
                  created=time.time())
    write(output / 'qualification.json', result)
    write(output / 'CPU_execution.json', dict(kind='PURE_CAMERA_GROUPED_SEMANTIC_QUALIFICATION_CPU_V1',
          frame=frame, workers=workers, native_threads_per_worker=1,
          core_source=str(core_source), core_source_sha256=sha(core_source),
          helper_sha256=sha(__file__), bindings_sha256=digest,
          ROI_order_preserved=True, source_inputs_not_mutated=True,
          camera_timings=timings, elapsed_seconds=time.time() - started))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--frame', type=int, required=True)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    print(qualify(args.frame, args.bindings, args.root, args.output, args.workers, args.data_root))
