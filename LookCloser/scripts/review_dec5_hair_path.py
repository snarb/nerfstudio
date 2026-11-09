#!/usr/bin/env python3
"""Static-path coverage/source-switch diagnostics plus actual encoded-frame sheets.

Common motion comes from the frozen baseline, never from candidate RGB. These
are image/geometry consistency proxies, not tracked 3D strands or temporal GT.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np

from dec5_hair import project, unproject
from run_dec5_hair_campaign import atomic_json, sha


def remap(array, uv, nearest=False):
    # OpenCV remap's destination rows/cols must fit in signed 16 bits.
    source = array.astype(np.float32)
    values = []
    for begin in range(0, len(uv), 16384):
        part = uv[begin:begin+16384]
        values.append(cv2.remap(source, (part[:, 0]-.5).astype(np.float32)[:, None],
                         (part[:, 1]-.5).astype(np.float32)[:, None],
                         cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=-1)[:, 0])
    return np.concatenate(values) if values else np.empty(0, np.float32)


def pair_metrics(previous, following, yy, xx, uv, z, mask, predicted_depth=None):
    hp = previous['hair_coverage'] if 'hair_coverage' in previous else previous['hair_probability']
    hn = following['hair_coverage'] if 'hair_coverage' in following else following['hair_probability']
    present = (previous['source_id'][yy, xx] >= 0) & (hp[yy, xx] > .25)
    next_present = (remap(following['source_id'], uv, True) >= 0) & (remap(hn, uv, True) > .25)
    toggles = present != next_present
    switches = (previous['source_id'][yy, xx] != remap(following['source_id'], uv, True)) & present & next_present
    depth = remap(following['front_depth'], uv, True)
    tolerance = np.maximum(2*z/9000, .002)  # calibration-unit numerical proxy, not mm accuracy
    prediction = z if predicted_depth is None else predicted_depth
    exposure = present & ((depth <= 0) | (prediction <= 0) | (depth > prediction+tolerance))
    return dict(anchors=int(mask.sum()), visibility_toggles=int((toggles & mask).sum()),
                source_switches=int((switches & mask).sum()),
                depth_exposure_dropouts=int((exposure & mask).sum()),
                present_pixels=int((present & mask).sum()), next_present_pixels=int((next_present & mask).sum())), toggles & mask


def encode_and_decode(directory, output, box, frame_count=96):
    output.mkdir(parents=True, exist_ok=False)
    video = output/'video.mp4'
    command = [os.environ.get('FFMPEG_BINARY','ffmpeg'),'-hide_banner','-loglevel','error','-framerate','24','-i',str(directory/'novel_%04d.png'),
               '-vf','transpose=2','-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p',str(video)]
    encoded = subprocess.run(command, capture_output=True, text=True)
    library_override = None
    if (encoded.returncode == 127 and 'symbol lookup error' in encoded.stderr
            and Path('/usr/lib/x86_64-linux-gnu').is_dir()):
        # Some hosts shadow system mpg123 via /usr/local/lib. Scope the fix to
        # this encoder subprocess; never alter the Python/service environment.
        environment = os.environ.copy()
        library_override = '/usr/lib/x86_64-linux-gnu'
        environment['LD_LIBRARY_PATH'] = library_override
        encoded = subprocess.run(command, capture_output=True, text=True, env=environment)
    if encoded.returncode:
        raise RuntimeError('Video encoder failed: '+encoded.stderr[-2000:])
    cap = cv2.VideoCapture(str(video))
    images = []
    x0, y0, x1, y1 = box
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tile = cv2.resize(frame[y0:y1, x0:x1], (108, 192), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, str(len(images)), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
        images.append(tile)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if len(images) != frame_count or abs(fps-24) > .01:
        raise ValueError('Encoded video frame inventory/fps mismatch')
    sheets = []
    for start in range(0, frame_count, 32):
        group = list(images[start:start+32])
        group += [np.zeros_like(images[0]) for _ in range((-len(group)) % 8)]
        sheet = np.concatenate([np.concatenate(group[i:i+8], 1) for i in range(0, len(group), 8)], 0)
        path = output/('decoded_%03d.jpg'%start)
        cv2.imwrite(str(path), sheet)
        sheets.append(dict(path=str(path), sha256=sha(path)))
    return dict(video=str(video), video_sha256=sha(video), frames=frame_count, fps=fps, duration_seconds=frame_count/24, sheets=sheets,
                encoder_library_path_override=library_override,encoder_binary=command[0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--path', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-frames', type=int, default=96,
                   help='Explicitly label a shorter pilot; default final gate remains 96.')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frames = json.loads(args.path.read_text())['frames']
    raw_height,raw_width=int(frames[0]['h']),int(frames[0]['w'])
    if any((int(f['h']),int(f['w']))!=(raw_height,raw_width) for f in frames):
        raise ValueError('Path review requires one consistent raster')
    count = len(frames)
    if count != args.expected_frames or not 2 <= count <= 96:
        raise ValueError('Unexpected frozen path inventory')
    receipt = [json.loads((d/'complete.json').read_text()) for d in (args.baseline, args.candidate)]
    if any(len(r['records']) != count for r in receipt):
        raise ValueError('Incomplete path render')
    rows, boxes, coverage_rows = [], [], []
    previous_b, previous_c = None, None
    worst = []
    for index, camera in enumerate(frames):
        name = camera['physical_camera']
        b = dict(np.load(args.baseline/(name+'.npz')))
        c = dict(np.load(args.candidate/(name+'.npz')))
        reference_hair=b['roi']&(b['hair_probability']>.25)
        coverage={}
        for label,values in [('baseline',b),('candidate',c)]:
            domain=values['roi'];present=(values['source_id']>=0)&(values['hair_probability']>.25)
            coverage[label]=dict(domain_pixels=int(domain.sum()),
                hair_present_pixels=int((domain&present).sum()),
                foreground_depth_pixels=int((domain&(values['front_depth']>0)).sum()),
                fallback_pixels=int((domain&(values['source_id']<0)).sum()),
                baseline_hair_domain_pixels=int(reference_hair.sum()),
                present_in_baseline_hair=int((reference_hair&present).sum()))
        coverage_rows.append(dict(frame=index,camera=name,**coverage))
        portrait_roi = np.rot90(b['roi'])
        yy0, xx0 = np.nonzero(portrait_roi)
        if len(xx0):
            boxes.append([xx0.min(), yy0.min(), xx0.max()+1, yy0.max()+1])
        if previous_b is not None:
            yy, xx = np.nonzero(previous_b['roi'])
            depth = previous_b['front_depth'][yy, xx]
            xyz = unproject(np.stack((xx+.5, yy+.5), -1), depth, frames[index-1])
            uv, z = project(xyz, camera)
            ref_next = remap(b['front_depth'], uv, True)
            tolerance = np.maximum(2*z/min(camera['fl_x'],camera['fl_y']), .002)
            mask = ((depth > 0) & (z > 0) & (uv[:, 0] >= .5) & (uv[:, 0] <= camera['w']-.5)
                    & (uv[:, 1] >= .5) & (uv[:, 1] <= camera['h']-.5)
                    & ~((ref_next > 0) & (ref_next < z-tolerance)))
            rb, tb = pair_metrics(previous_b, b, yy, xx, uv, z, mask)
            # Transport candidate previous depth too: a systematic offset from
            # B0 is geometry disagreement, not a frame-to-frame depth dropout.
            candidate_points = unproject(np.stack((xx+.5, yy+.5), -1), previous_c['front_depth'][yy, xx], frames[index-1])
            _, candidate_z = project(candidate_points, camera)
            candidate_z = np.where(previous_c['front_depth'][yy, xx] > 0, candidate_z, 0)
            rc, tc = pair_metrics(previous_c, c, yy, xx, uv, z, mask, candidate_z)
            rows.append(dict(pair=[index-1,index], baseline=rb, candidate=rc,
                             unknown_reference_pixels=int((depth <= 0).sum()), occlusion_policy='frozen baseline near-surface veto'))
            flip = np.zeros(previous_b['roi'].shape, np.uint8)
            flip[yy[tb | tc], xx[tb | tc]] = 1
            smoothed = cv2.GaussianBlur(flip.astype(np.float32), (0,0), 8)
            y, x = np.unravel_index(smoothed.argmax(), smoothed.shape)
            worst.append((int((tb | tc).sum()), index-1, int(x), int(y)))
        previous_b, previous_c = b, c
        atomic_json(args.output/'progress.json', dict(status='scoring', completed=index+1, total=count))
    aggregates = {}
    for key in ('baseline','candidate'):
        totals = {field:sum(r[key][field] for r in rows) for field in rows[0][key]}
        denominator = max(totals['anchors'], 1)
        aggregates[key] = dict(totals, **{field+'_fraction':totals[field]/denominator
                              for field in ('visibility_toggles','source_switches','depth_exposure_dropouts','present_pixels')})
    boxes = np.array(boxes)
    box = [max(int(boxes[:,0].min())-32, 0), max(int(boxes[:,1].min())-32, 0),
           min(int(boxes[:,2].max())+32, raw_height), min(int(boxes[:,3].max())+192, raw_width)]
    videos = dict(baseline=encode_and_decode(args.baseline, args.output/'baseline', box, count),
                  candidate=encode_and_decode(args.candidate, args.output/'candidate', box, count))
    patches = []
    for rank, (_, index, x, y) in enumerate(sorted(worst, reverse=True)[:3]):
        x0, x1 = max(0,x-80), min(raw_width,x+80)
        y0, y1 = max(0,y-80), min(raw_height,y+80)
        tiles = []
        for directory in (args.baseline, args.candidate):
            for i in (index,index+1):
                image = cv2.imread(str(directory/('novel_%04d.png'%i)))
                tiles.append(np.rot90(image[y0:y1,x0:x1]))
        height = min(t.shape[0] for t in tiles)
        panel = np.concatenate([t[:height] for t in tiles], 1)
        path = args.output/('problem_patch_%d.jpg'%rank)
        cv2.imwrite(str(path), panel)
        patches.append(dict(path=str(path), frames=[index,index+1], original_box=[x0,y0,x1,y1], sha256=sha(path)))
    atomic_json(args.output/'metrics.json', dict(aggregate=aggregates, per_pair=rows,
        unconditional_per_frame_coverage=coverage_rows,
        coverage_domain='whole fixed crop plus baseline train-voted hair; no GT topology claim',
        path_sha256=sha(args.path), videos=videos, problem_patches=patches,
        baseline_receipt_sha256=sha(args.baseline/'complete.json'),
        candidate_receipt_sha256=sha(args.candidate/'complete.json'),
        camera_path_only=True, physical_time_stability_established=False,
        frame_count=count, pilot=count < 96,
        depth_transport='candidate previous depth; frozen baseline image correspondences/domain/occlusion veto',
        metric_kind='fixed-baseline warped coverage and source-switch proxies, no strand GT',
        missing_candidate_pixels_excluded=False, human_continuous_playback_claimed=False))
    atomic_json(args.output/'progress.json', dict(status='complete', completed=count, total=count))


if __name__ == '__main__':
    main()
