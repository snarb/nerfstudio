"""Two-channel 2D guidance; current NN components are proposals, never foreground.

Individual identity cores exclude ambiguous ownership. The separate complete
actor union retains it. No previous bitmap, model, grid or weight is transferred.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import fcntl

import cv2
import numpy as np
from PIL import Image

SIZE = (1024, 540)
NATIVE = (4320, 8192)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.partial.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)

def pool(mask):
    """Exact ANY pooling on the native8x8 footprint; no thin union loss."""
    mask = np.asarray(mask, bool)
    if mask.shape != NATIVE:
        raise ValueError('Expected original RAW native4320x8192 mask')
    return mask.reshape(540, 8, 1024, 8).any(axis=(1, 3))

def channels(black, grey, uncertainty, veto):
    b = np.asarray(black, bool) & ~veto
    g = np.asarray(grey, bool) & ~veto
    return {'identity_black': pool(b & ~uncertainty),
            'identity_grey': pool(g & ~uncertainty),
            'complete_union': pool(b | g)}

def flow_channels(before, current, source):
    """Same Farneback/FB/photo thresholds as the current read-only source guard."""
    forward = cv2.calcOpticalFlowFarneback(before, current, None, .5, 3, 21, 3, 7, 1.5, 0)
    backward = cv2.calcOpticalFlowFarneback(current, before, None, .5, 3, 21, 3, 7, 1.5, 0)
    yy, xx = np.indices(before.shape, dtype=np.float32)
    tx, ty = xx + forward[..., 0], yy + forward[..., 1]
    bx = cv2.remap(backward[..., 0], tx, ty, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    by = cv2.remap(backward[..., 1], tx, ty, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    fb = np.hypot(forward[..., 0] + bx, forward[..., 1] + by)
    photo = cv2.remap(current, tx, ty, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    reliable = (fb < 1.5) & (np.abs(photo.astype(np.float32) - before.astype(np.float32)) < 35)
    reliable &= (tx >= 0) & (tx < SIZE[0]) & (ty >= 0) & (ty < SIZE[1])
    result = {}
    for key, mask in source.items():
        # Only per-person ID cores are eroded; full union must retain ownership-U.
        support = mask if key == 'complete_union' else cv2.erode(mask.astype('uint8'), np.ones((3, 3), 'uint8')) > 0
        good = support & reliable
        out = np.zeros(before.shape, bool)
        out[np.rint(ty[good]).astype(int).clip(0, SIZE[1] - 1),
            np.rint(tx[good]).astype(int).clip(0, SIZE[0] - 1)] = True
        result['flowed_' + key] = out
    result['flow_reliable'] = reliable
    return result

def native_inputs(root, frame, camera, native=None):
    path = Path(native) if native else root / 'frames' / f'{frame:06d}' / 'native_masks' / (camera + '.npz')
    metadata = read(path.with_suffix('.json'))
    actual = sha(path)
    if metadata.get('frame') != frame or metadata.get('camera') != camera or metadata.get('archive_sha256') != actual:
        raise ValueError('Native mask metadata/source binding differs')
    with np.load(path) as z:
        arrays = {k: z[k].copy() for k in ['black', 'grey', 'source_ineligible', 'unknown']}
        arrays['actor_ownership_uncertain'] = z['actor_ownership_uncertain'].copy() if 'actor_ownership_uncertain' in z.files else np.zeros(NATIVE, bool)
        arrays['qualified_actor_rgb'] = z['qualified_actor_rgb'].copy() if 'qualified_actor_rgb' in z.files else np.zeros(NATIVE, bool)
    return path, actual, arrays

def raw_input(root, frame, camera):
    receipt = root / 'sources' / str(frame) / camera / 'receipt.json'
    row = read(receipt)
    image = Path(row['raw_native_image'])
    digest = sha(image)
    if row['frame'] != frame or row['camera'] != camera or digest != row['raw_native_image_sha256']:
        raise ValueError('Current RAW raster/frame/source receipt changed')
    return image, {'receipt_sha256': sha(receipt), 'raw_native_image_sha256': digest}

def candidate_input(root, frame, camera):
    for folder in ['person_candidates_sg', 'person_candidates']:
        p = root / 'frames' / f'{frame:06d}' / folder / (camera + '.npz')
        if p.exists():
            row = read(p.with_suffix('.json'))
            digest = sha(p)
            original = row.get('original_source_receipt_sha256', row.get('source_receipt_sha256'))
            if row.get('frame') != frame or row.get('camera') != camera or digest != row['archive_sha256'] or original != sha(root / 'sources' / str(frame) / camera / 'receipt.json'):
                raise ValueError('Current NN candidate archive/source differs')
            return p, digest
    raise FileNotFoundError('Missing fresh current-frame NN candidates: ' + camera)

def build_channels(root, frame, camera, output, previous=None):
    """Cache source-bound guidance arrays; independent of current mask publication."""
    root, output = Path(root), Path(output)
    previous = frame - 5 if previous is None else previous
    if not 0 < frame - previous <= 60:
        raise ValueError('Previous2D guidance must be earlier and within60 source frames')
    output.mkdir(parents=True, exist_ok=True)
    prevpath, prevsha, old = native_inputs(root, previous, camera)
    old_image, old_binding = raw_input(root, previous, camera)
    current_image, current_binding = raw_input(root, frame, camera)
    binding = dict(kind='TWO_CHANNEL_2D_UNION_FLOW_GUIDANCE_V1', frame=frame, previous=previous,
                   camera=camera, previous_native_sha256=prevsha,
                   previous_native_metadata_sha256=sha(prevpath.with_suffix('.json')),
                   previous_RAW=old_binding, current_RAW=current_binding, code_sha256=sha(__file__),
                   complete_union_includes_ownership_uncertain=True,
                   identity_cores_exclude_ownership_uncertain=True,
                   no_previous_bitmap_supplied_as_final_foreground=True,
                   no_NeRF_state_loaded=True, guidance_only=True)
    path, proof = output / (camera + '_channels.npz'), output / (camera + '_channels.json')
    if path.exists() and proof.exists():
        saved = read(proof)
        if saved.get('binding') == binding and saved.get('archive_sha256') == sha(path):
            with np.load(path) as z:
                return {k: z[k].copy() for k in z.files}, saved
    original = channels(old['black'], old['grey'], old['actor_ownership_uncertain'], old['source_ineligible'])
    images = [np.asarray(Image.open(p).convert('L').resize(SIZE, Image.Resampling.BILINEAR)) for p in [old_image, current_image]]
    arrays = dict(original)
    arrays.update(flow_channels(images[0], images[1], original))
    arrays['flowed_union_envelope128'] = cv2.dilate(arrays['flowed_complete_union'].astype('uint8'), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))) > 0
    arrays['previous_ownership_uncertain_coarse'] = pool(old['actor_ownership_uncertain'] & (old['black'] | old['grey']) & ~old['source_ineligible'])
    tmp = path.with_name(path.name + '.partial.' + str(os.getpid()))
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)
    if sha(prevpath) != prevsha or sha(old_image) != old_binding['raw_native_image_sha256'] or sha(current_image) != current_binding['raw_native_image_sha256']:
        raise ValueError('Guidance input changed while computing flow')
    result = {'binding': binding, 'archive_sha256': sha(path), 'array_sha256': {k: array_sha(v) for k, v in arrays.items()},
              'pixels': {k: int(v.sum()) for k, v in arrays.items()}, 'created': time.time(),
              'thresholds': {'FB_pixels': 1.5, 'photometric_uint8': 35, 'full_union_erosion': 0, 'identity_core_erosion': 1}}
    atomic(proof, result)
    return arrays, result

def _crew_context(root, frame, camera, current_masks, candidate_sha, qualified_actor_rgb):
    report = root / 'frames' / f'{frame:06d}' / 'crew_guard' / (camera + '_people.json')
    if not report.exists():
        return {}, {}, {'available': False}
    row = read(report)
    if row.get('frame') != frame or row.get('camera') != camera or row.get('current_candidate_archive_sha256') != candidate_sha:
        return {}, {}, {'available': False, 'guard_sha256': sha(report), 'reason': 'Current crew report has different NN/frame binding'}
    previous = frame - 5
    try:
        previous_path, previous_sha = candidate_input(root, previous, camera)
        with np.load(previous_path) as z:
            old_masks = [pool(m) for m in z['masks']]
        previous_native = root / 'frames' / f'{previous:06d}' / 'native_masks' / (camera + '.npz')
        with np.load(previous_native) as z:
            previous_rgb = pool(z['qualified_actor_rgb']) if 'qualified_actor_rgb' in z.files else np.zeros((540, 1024), bool)
        # Match the source guard: qualified current/prior actor pixels cannot
        # become head-connected CREW anchors merely through a mixed NN instance.
        old_masks = [m & ~previous_rgb for m in old_masks]
        helper = root / 'operations' / 'marshawn_actor_conflict_v51.py'
        for p in [root / 'operations', root / 'code/LookCloser/scripts', root / 'code/scripts']:
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        spec = importlib.util.spec_from_file_location('union_tracking_current_crew_helper', helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cores, proofs = {}, {}
        for person in row['people']:
            match = person.get('previous_identity_match')
            if not match or not 0 <= match.get('instance', -1) < len(old_masks):
                continue
            i = person['instance']
            core, proof = module.qualified_current_crew_core(current_masks[i] & ~qualified_actor_rgb, old_masks[match['instance']], match)
            cores[i], proofs[i] = core, proof
        return cores, proofs, {'available': True, 'guard_sha256': sha(report), 'previous_frame': previous,
                              'previous_candidate_sha256': previous_sha, 'previous_native_sha256': sha(previous_native),
                              'current_qualified_actor_rgb_sha256': array_sha(qualified_actor_rgb),
                              'crew_helper_sha256': sha(helper)}
    except FileNotFoundError:
        return {}, {}, {'available': False, 'guard_sha256': sha(report), 'reason': 'Previous current-NN crew context unavailable'}

def propose(root, frame, camera, output, previous=None, current_native=None):
    """Return current native-component ROI proposals, never qualified mask pixels."""
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / (camera + '.lock')).open('a+') as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        arrays, guidance = build_channels(root, frame, camera, output, previous)
        native, native_sha, current = native_inputs(root, frame, camera, current_native)
        candidate, candidate_sha = candidate_input(root, frame, camera)
        with np.load(candidate) as z:
            masks = z['masks'].copy()
        union = pool((current['black'] | current['grey']) & ~current['source_ineligible'])
        missing = arrays['flowed_complete_union'] & ~union
        near_missing = arrays['flowed_union_envelope128'] & ~union
        crew_cores, crew_proofs, crew_binding = _crew_context(root, frame, camera, masks, candidate_sha, current['qualified_actor_rgb'])
        records = []
        for instance, mask in enumerate(masks):
            coarse = pool(mask)
            if int((coarse & missing).sum()) < 4 and int((coarse & near_missing).sum()) < 8:
                continue
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'), connectivity=8)
            for component in range(1, count):
                x, y, w, h, area = map(int, stats[component])
                if area < 64:
                    continue
                cx0, cy0, cx1, cy1 = x // 8, y // 8, (x + w + 7) // 8, (y + h + 7) // 8
                part = labels[cy0 * 8:cy1 * 8, cx0 * 8:cx1 * 8] == component
                part = part.reshape(cy1 - cy0, 8, cx1 - cx0, 8).any(axis=(1, 3))
                sl = (slice(cy0, cy1), slice(cx0, cx1))
                direct, nearby = int((part & missing[sl]).sum()), int((part & near_missing[sl]).sum())
                if direct < 4 and nearby < 8:
                    continue
                crew = crew_proofs.get(instance, {'qualified': False})
                crew_overlap = int((part & crew_cores[instance][sl]).sum()) if crew.get('qualified') else None
                native_sl = (slice(y, y + h), slice(x, x + w))
                native_part = labels[native_sl] == component
                actor_pixels = int((native_part & (current['black'][native_sl] | current['grey'][native_sl])).sum())
                records.append({'instance': instance, 'native_component': component, 'native_pixels': area,
                    'bbox_native_xyxy': [x, y, x + w, y + h],
                    'ROI_native_xyxy': [max(0, x - 128), max(0, y - 128), min(8192, x + w + 128), min(4320, y + h + 128)],
                    'flowed_complete_union_pixels_coarse': int((part & arrays['flowed_complete_union'][sl]).sum()),
                    'missing_current_union_flow_pixels_coarse': direct,
                    'missing_current_union_envelope128_pixels_coarse': nearby,
                    'direct_flow_anchor_present': direct >= 4,
                    'current_actor_pixels_native': actor_pixels,
                    'current_source_ineligible_pixels_native': int((native_part & current['source_ineligible'][native_sl]).sum()),
                    'current_geometry_UNKNOWN_pixels_native': int((native_part & current['unknown'][native_sl]).sum()),
                    'identity_black_flow_pixels_coarse': int((part & arrays['flowed_identity_black'][sl]).sum()),
                    'identity_grey_flow_pixels_coarse': int((part & arrays['flowed_identity_grey'][sl]).sum()),
                    'current_qualified_crew_head_connected_overlap_pixels_coarse': crew_overlap,
                    'current_crew_head_core_proof': crew,
                    'status': 'CURRENT_NN_COMPONENT_ROI_PROPOSAL_REQUIRES_FRESH_SEMANTIC_CONSENSUS',
                    'envelope_only_is_not_actor_identity': direct < 4})
        if sha(native) != native_sha or sha(candidate) != candidate_sha:
            raise ValueError('Current input changed during proposal computation')
        result = {'kind': 'CURRENT_NATIVE_COMPONENT_TWO_CHANNEL_ROI_PROPOSALS_V1', 'frame': frame, 'camera': camera,
            'previous': frame - 5 if previous is None else previous, 'guidance_sha256': sha(output / (camera + '_channels.json')),
            'guidance': guidance, 'current_native_sha256': native_sha, 'current_candidate_sha256': candidate_sha,
            'crew_context': crew_binding, 'records': records, 'no_final_foreground_or_loss_weights_emitted': True,
            'no_model_or_grid_state_loaded': True, 'current_NN_consensus_and_independent_crew_guard_required': True,
            'input_files_not_modified': True, 'created': time.time()}
        atomic(output / (camera + '_proposals.json'), result)
        return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--frame', required=True, type=int)
    p.add_argument('--previous', type=int)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--cameras', nargs='+', required=True)
    p.add_argument('--channels-only', action='store_true')
    a = p.parse_args()
    for camera in a.cameras:
        if a.channels_only:
            _, r = build_channels(a.root, a.frame, camera, a.output, a.previous)
            print(json.dumps({'camera': camera, 'channels': r['pixels']}), flush=True)
        else:
            r = propose(a.root, a.frame, camera, a.output, a.previous)
            print(json.dumps({'camera': camera, 'proposals': len(r['records'])}), flush=True)

if __name__ == '__main__':
    main()
