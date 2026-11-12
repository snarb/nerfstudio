"""Motion-align field-derived support across three frames without blending RGB."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy import ndimage as ndi
from render_depth_cleanup import depth_support


def main(root, name='temporal_support', minimum_component=2000, preserve_lower_border=False, minimum_height=0):
    cv2.setNumThreads(1)
    root = Path(root)
    frames = list(range(908, 1268, 5))
    unit = .2217203609534452*1.0805

    def prepare(frame):
        folder = root/f'{frame:06}'
        with np.load(folder/'raw.npz') as raw:
            core, distance, reference = depth_support(raw, minimum_component, preserve_lower_border, minimum_height)
            signed = distance - ndi.distance_transform_edt(core).astype(np.float32)
            image = np.uint8(np.clip(np.sqrt(np.maximum(raw['rgb'], 0))*255, 0, 255))
            grey = cv2.cvtColor(cv2.resize(image, None, fx=.5, fy=.5), cv2.COLOR_RGB2GRAY)
        return frame, dict(signed=signed, reference=reference, grey=grey)

    with ThreadPoolExecutor(8) as pool:
        base = dict(pool.map(prepare, frames))

    def stabilize(frame):
        row = base[frame]
        h, w = row['signed'].shape
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        candidates = [row['signed']]
        valid_counts = []
        for neighbour in (frame-5, frame+5):
            if neighbour not in base:
                continue
            other = base[neighbour]
            estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            forward = estimator.calc(row['grey'], other['grey'], None)
            backward = estimator.calc(other['grey'], row['grey'], None)
            forward = cv2.resize(forward, (w, h))*2
            backward = cv2.resize(backward, (w, h))*2
            mx, my = xx+forward[..., 0], yy+forward[..., 1]
            def warp(value):
                return cv2.remap(value, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            reverse = warp(backward)
            error = np.linalg.norm(forward+reverse, axis=-1)
            valid = (mx >= 0)&(mx < w-1)&(my >= 0)&(my < h-1)&(error < 2)
            valid &= np.abs(warp(other['reference'])-row['reference']) < .20*unit
            valid &= (row['signed'] > -6)&(row['signed'] < 30)
            candidates.append(np.where(valid, warp(other['signed']), row['signed']))
            valid_counts.append(int(valid.sum()))
        median = np.median(np.stack(candidates), axis=0)
        # Limit any contour movement to two pixels; do not temporal-average color.
        stabilized = row['signed'] + np.clip(median-row['signed'], -2, 2)
        distance = np.maximum(stabilized, 0).astype(np.float32)
        np.savez_compressed(root/f'{frame:06}'/(name+'.npz'),
                            distance=distance, reference=row['reference'], minimum_component=minimum_component,
                            minimum_height=minimum_height)
        return dict(frame=frame, valid_motion_pixels=valid_counts,
                    max_support_shift_pixels=float(np.max(np.abs(stabilized-row['signed']))))

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(stabilize, frames))
    receipt = dict(frames=72, core_alpha=.98, minimum_component=minimum_component,
        preserve_lower_border=preserve_lower_border,
        minimum_height_pixels=minimum_height,
        temporal_method='motion-aligned median signed distance, at most2px',
        forward_backward_max_error_pixels=2, depth_agreement_metres=.20,
        RGB_blending=False, SAM_masks_used=False, rows=rows)
    (root/(name+'_receipt.json')).write_text(json.dumps(receipt, indent=2)+'\n')
    print('TEMPORAL_SUPPORT_DONE', len(rows))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--name',default='temporal_support');p.add_argument('--minimum-component',type=int,default=2000)
    p.add_argument('--preserve-lower-border',action='store_true')
    p.add_argument('--minimum-height',type=int,default=0)
    a=p.parse_args();main(a.output,a.name,a.minimum_component,a.preserve_lower_border,a.minimum_height)
