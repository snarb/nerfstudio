"""Independent checks for the three-time held-out color experiment."""
from pathlib import Path
import numpy as np
from PIL import Image
from joint_temporal_texture import read, sha, atomic_json, HELD_CAMERAS
from render_smooth_temporal_mesh_video import verify_request
from score_colmap_patchmatch_tsdf_face import load_manual_face_mask
from audit_bounded_temporal_gradient_control import audit


def run(root):
    request=verify_request(root); rows=[]
    if request['ordered_frame_ids']!=['000899','000973','001193']:
        raise ValueError('Wrong held-out frame inventory')
    for record in request['inventory']:
        frame=record['frame_id']; base=root/'frames'/frame; ev=root/'evaluation'/frame
        receipt=read(base/'complete.json')
        if receipt['request_sha256']!=sha(root/'request.json'):raise ValueError('Render request changed')
        for name,digest in receipt['hashes'].items():
            if sha(base/name)!=digest:raise ValueError('Render output changed')
        result=read(base/'result.json')
        if len(result['source_cameras'])!=62 or set(result['source_cameras'])&HELD_CAMERAS or result['target_rgb_read']:
            raise ValueError('Held-out RGB source leakage')
        if result['graph']['target_angle_sigma_degrees']!=request['recipe']['target_angle_sigma_degrees']:
            raise ValueError('Wrong production source prior')
        if sha(record['mesh'])!=record['mesh_sha256']:raise ValueError('Changed geometry')
        audit(root/'correction'/frame/'bounded')
        metric=read(ev/'metrics.json'); gt=np.array(Image.open(ev/'gt.png'))/255
        gt_receipt=read(ev/'gt_receipt.json')
        if sha(gt_receipt['source'])!=gt_receipt['source_sha256']:
            raise ValueError('Held-out source changed')
        if [r['variant'] for r in metric['rows']]!=['baseline','bounded']:
            raise ValueError('Wrong comparison inventory')
        mask,_=load_manual_face_mask(ev/'roi.json',ev/'gt.png',gt.shape[:2])
        if metric['full_frame_metrics'] or metric['heldout_rgb_used_for_prediction']:
            raise ValueError('Unexpected metric scope or leakage')
        if sha(ev/'roi.json')!=metric['roi_sha256'] or sha(ev/'gt.png')!=metric['gt_sha256']:
            raise ValueError('Changed GT/ROI')
        for r in metric['rows']:
            path=base/'frame.png' if r['variant']=='baseline' else root/'correction'/frame/'bounded/corrected.png'
            if sha(path)!=r['prediction_sha256']:raise ValueError('Changed metric prediction')
            pred=np.array(Image.open(path))/255
            psnr=-10*np.log10(max(np.square(pred[mask]-gt[mask]).mean(),1e-12))
            if abs(psnr-r['face_psnr'])>1e-4:raise ValueError('Independent face PSNR mismatch')
            if not all(np.isfinite(r[k]) for k in ['face_psnr','face_ssim','face_lpips']):
                raise ValueError('Nonfinite face score')
        review=read(ev/'visual_review.json')
        if sha(ev/'native_face_hair_comparison.png')!=review['comparison_sha256'] or review['actually_viewed_native'] is not True:
            raise ValueError('Unbound visual review')
        rows.append(dict(frame=frame,metric_sha256=sha(ev/'metrics.json'),review_sha256=sha(ev/'visual_review.json'),
                         face_psnr_independently_verified=True))
    atomic_json(root/'audit.json',dict(rows=rows,request_sha256=sha(root/'request.json'),script_sha256=sha(__file__),
        compared_frames=3,metric_triplets=6,full_frame_metrics=False,production_promoted=False,
        decision='reject global low-ridge color correction: color drift despite seam improvement'))
    print('Audited 3 frames / 6 face-only metric triplets; candidate not promoted',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/mnt/data/dec5_gradient_heldout_fidelity'))
    run(p.parse_args().root)
