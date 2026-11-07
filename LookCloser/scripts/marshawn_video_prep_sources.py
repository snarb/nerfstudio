"""Frame-local video RAW decode; select the qualified original or rebuilt tail.

Decode specified temporal samples with the frozen 1660 colour/camera contract.

HD previews only; segmentation must use separately requested native crops.
No frame-dependent exposure or pose estimation is performed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import tifffile
from PIL import Image

from prepare_marshawn_color_variants import vendor, RAW, sha, srgb
from marshawn_actor_coordinates import maps, raw_camera, target_intrinsics
from marshawn_radial_photometry import radial_ev

ROOT = Path('/fsx/tmp/marshawn_001660_hd')


def main(a):
    cv2.setNumThreads(1)
    tf_path = ROOT/'color_variants/radial_k4_joint_scalar_DIAGNOSTIC/sdr_png/transforms.json'
    fit_path = ROOT/'report/shared_radial1660_train_PRECHECK/fit_k4.json'
    tf = json.loads(tf_path.read_text())
    fit = json.loads(fit_path.read_text())
    pq = vendor()
    matrix = pq.rgb_to_rgb_matrix('rwg', 'rec709')
    rows = {r['physical_camera']: r for r in tf['frames']}
    if set(a.cameras)-set(rows):raise ValueError('Unknown calibrated cameras: '+str(set(a.cameras)-set(rows)))
    rois = json.loads(a.native_rois.read_text()) if a.native_rois else None
    a.output.mkdir(parents=True, exist_ok=True)

    def one(task):
        frame, name = task
        out = a.output/str(frame)/name
        out.mkdir(parents=True, exist_ok=True)
        row = rows[name]
        desired=None
        if rois is not None:
            scaled=np.asarray(rois[name],float)*np.tile([8192/row['w'],4320/row['h']],2)
            desired=np.r_[np.maximum(np.floor(scaled[:2]),0),
                          np.minimum(np.ceil(scaled[2:]),[8192,4320])].astype(int).tolist()
        if (out/'receipt.json').exists():
            prior=json.loads((out/'receipt.json').read_text())
            same=(prior['frame']==frame and prior['camera']==name
                  and prior['transforms_sha256']==sha(tf_path) and prior['photometry_sha256']==sha(fit_path)
                  and Path(prior['image']).exists() and sha(prior['image'])==prior['image_sha256'])
            native_ok=(rois is None or (prior.get('native_crop_xyxy')==desired
                and prior.get('native_image') and Path(prior['native_image']).exists()
                and sha(prior['native_image'])==prior['native_image_sha256']))
            raw_ok=(not a.raw_native or (prior.get('raw_native_image') and Path(prior['raw_native_image']).exists() and sha(prior['raw_native_image'])==prior['raw_native_image_sha256']))
            if same and native_ok and raw_ok:
                return prior
        source = (RAW if frame<=1900 else Path('/fsx/oregon/gta_marshawn_2_1_tiff16_f1901-2020_20261105/by_local'))/name/('%s.%06d.tif' % (name, frame-1252))
        raw = tifffile.memmap(source, mode='r')
        assert raw.shape == (4320,8192,3) and raw.dtype == np.uint16
        linear = np.empty(raw.shape, np.float32)
        for y in range(0,4320,128):
            linear[y:y+128] = np.maximum(np.maximum(pq.log3g10_decode(
                raw[y:y+128].astype(np.float64)/65535),0) @ matrix.T,0)
        mx,my = maps(row)
        small = cv2.resize(linear,(row['w'],row['h']),interpolation=cv2.INTER_AREA)
        hd = cv2.remap(small,(mx+.5)*(row['w']/8192)-.5,
                       (my+.5)*(row['h']/4320)-.5,cv2.INTER_LINEAR)
        ev = float(fit['eval_predictions'][name]['ev'] if name in fit['eval_predictions']
                   else next(r['scalar_correction_EV'] for r in fit['camera_rows'] if r['camera']==name))
        hd *= (4*np.exp2(radial_ev(row,4)-float(fit['radial_global_center_EV'])+ev))[...,None]
        image = out/'source.png'
        Image.fromarray(srgb(hd)).save(image)
        result = dict(frame=frame, time_seconds=(frame-1660)/120, camera=name,
                      raw=str(source), raw_sha256=sha(source), image=str(image),
                      image_sha256=sha(image), camera_row=row, fixed_EV=ev,
                      transforms_sha256=sha(tf_path), photometry_sha256=sha(fit_path),
                      script_sha256=sha(__file__), native_masks=False)
        if rois is not None:
            full = dict(row,**target_intrinsics(row))
            # Apply different x/y scale to both rectangle endpoints.
            crop = np.asarray(rois[name],float)*np.tile([8192/row['w'],4320/row['h']],2)
            x0,y0 = np.floor(crop[:2]).astype(int)
            x1,y1 = np.ceil(crop[2:]).astype(int)
            x0,y0 = max(0,x0),max(0,y0)
            x1,y1 = min(8192,x1),min(4320,y1)
            K,d = raw_camera(name)
            dst = np.array([[full['fl_x'],0,full['cx']-x0-.5],
                            [0,full['fl_y'],full['cy']-y0-.5],[0,0,1.]])
            nx,ny = cv2.initUndistortRectifyMap(K,d,None,dst,(x1-x0,y1-y0),cv2.CV_32FC1)
            native = cv2.remap(linear,nx,ny,cv2.INTER_LINEAR)
            Y,X = np.ogrid[y0:y1,x0:x1]
            uv = np.stack(np.broadcast_arrays(X+.5,Y+.5),axis=-1)
            native *= (4*np.exp2(radial_ev(full,4,uv)-float(fit['radial_global_center_EV'])+ev))[...,None]
            npng = out/'native.png'
            Image.fromarray(srgb(native)).save(npng,compress_level=1)
            result.update(native_image=str(npng),native_image_sha256=sha(npng),
                native_crop_xyxy=[int(x) for x in [x0,y0,x1,y1]],
                native_camera=dict(full,w=int(x1-x0),h=int(y1-y0),cx=float(full['cx']-x0),cy=float(full['cy']-y0)),
                native_filter='Native linear lens-remap; frozen gain4 joint k4 scalar EV; Reinhard sRGB',
                ROI_is_search_scope_not_actor_mask=True)
        if a.raw_native:
            # Segmentation-only RAW-raster image: global frozen gain/EV without
            # fitting new colour. Training source.png above keeps exact k4 chain.
            # Native masks use this distorted raster and the baseline AREA+remap.
            rawpng=out/'raw_native.png'
            Image.fromarray(srgb(linear*(4*np.exp2(ev)))).save(rawpng,compress_level=1)
            result.update(raw_native_image=str(rawpng),raw_native_image_sha256=sha(rawpng),raw_native_raster=[8192,4320],raw_native_purpose='SAM3 masks only; trainingRGB source.png exact frozenphotometry; no calibration resample',raw_native_photometry='Log3G10/RWG to linear Rec709; frozen4x cameraEV; Reinhard+sRGB')
        del linear
        (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
        print(frame,name,flush=True)
        return result

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(one,[(f,c) for f in a.frames for c in a.cameras]))
    import hashlib
    batch=hashlib.sha256(json.dumps([a.frames,a.cameras,str(a.native_rois)]).encode()).hexdigest()[:12]
    (a.output/('batch_'+batch+'.json')).write_text(json.dumps(dict(records=results,
        record_fps=120, output_fps=24, optimizer_updates=0),indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--frames',type=int,nargs='+',default=[1660,1720,1780,1840,1900])
    p.add_argument('--cameras',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--raw-native',action='store_true',help='Export native RAW-raster segmentation input; never a training target')
    p.add_argument('--native-rois',type=Path,help='Per-camera HD rectangles defining native search crops')
    main(p.parse_args())
