"""Evaluation-only motion-compensated RGB-change proxies for sealed sequences.

GT optical flow never enters training or checkpoint selection. These proxies
measure changes in RGB error, not ground-truth hair topology or metric depth.
"""
from pathlib import Path
import hashlib
import cv2
import numpy as np
from run_dec5_sequence import read, write


def residual_metrics(previous, current, gt_previous, gt_current, forward, backward, region=None):
    h,w=forward.shape[:2]
    y,x=np.mgrid[:h,:w].astype(np.float32)
    mx=x+forward[...,0];my=y+forward[...,1]
    def warp(values, interpolation=cv2.INTER_LINEAR):
        return cv2.remap(values,mx,my,interpolation,borderMode=cv2.BORDER_CONSTANT)
    roundtrip=forward+warp(backward)
    valid=(mx>=1)&(mx<w-2)&(my>=1)&(my<h-2)&(np.linalg.norm(roundtrip,axis=-1)<=1.5)
    if region is not None:valid &= region.astype(bool)
    count=int(valid.sum())
    if not count:return dict(valid_pixels=0,error_change_mae=None,prediction_change_mae=None,gt_change_mae=None)
    pred_delta=warp(current)-previous
    gt_delta=warp(gt_current)-gt_previous
    return dict(valid_pixels=count,
        error_change_mae=float(np.abs(pred_delta-gt_delta)[valid].mean()),
        prediction_change_mae=float(np.abs(pred_delta)[valid].mean()),
        gt_change_mae=float(np.abs(gt_delta)[valid].mean()))


def measure(root,manifest,results,output,reference=None):
    """Stream native RGB down to960×480; keep official checkpoint choices intact."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(2)
    size=(960,480);mode_rows={mode:result['per_frame'] for mode,result in results.items()}
    frame_names=manifest['frames'];previous=None;previous_gt=None;rows=[]
    reference_mask=None
    reference_sha=None
    if reference is not None:
        reference_sha=hashlib.sha256(Path(reference).read_bytes()).hexdigest()
        probability=np.load(reference)['probability'].astype(np.float32)/65535
        reference_mask=cv2.resize((probability[...,0]>.5).astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)>0
    hair_mask=reference_mask
    for index,frame in enumerate(frame_names):
        directory=root/'frames'/frame;data=directory/'data';meta=read(data/'transforms.json')
        eval_names=set(meta['val_filenames']);heldout=[f for f in meta['frames'] if f['file_path'] in eval_names]
        if len(heldout)!=1 or heldout[0]['physical_camera']!=manifest['heldout_camera']:
            raise ValueError('Temporal review requires the same held-out camera')
        image=cv2.imread(str(data/heldout[0]['file_path']))
        if image is None:raise ValueError('Missing sealed GT frame')
        gt=cv2.resize(image[...,::-1],size,interpolation=cv2.INTER_AREA).astype(np.float32)/255
        predictions={}
        for mode,records in mode_rows.items():
            if records[index]['frame']!=frame:raise ValueError('Temporal frame order differs')
            with np.load(Path(records[index]['renders'])/'0.npz') as archive:
                rgb=archive['rgb']
                if rgb.shape!=(3072,6144,3) or not np.isfinite(rgb).all():
                    raise ValueError('Invalid sealed native RGB')
                predictions[mode]=cv2.resize(rgb.astype(np.float32),size,interpolation=cv2.INTER_AREA)
        if previous is not None:
            old_gray=cv2.cvtColor(np.rint(previous_gt*255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
            gray=cv2.cvtColor(np.rint(gt*255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
            flow=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            forward=flow.calc(old_gray,gray,None)
            backward=flow.calc(gray,old_gray,None)
            regions=dict(full_frame=None)
            if hair_mask is not None:regions['propagated_hair_proxy']=hair_mask
            scores={mode:{name:residual_metrics(previous[mode],predictions[mode],previous_gt,gt,
                        forward,backward,mask) for name,mask in regions.items()} for mode in predictions}
            rows.append(dict(previous_frame=frame_names[index-1],frame=frame,metrics=scores))
            if hair_mask is not None:
                y,x=np.mgrid[:size[1],:size[0]].astype(np.float32)
                hair_mask=cv2.remap(hair_mask.astype(np.uint8),x+backward[...,0],y+backward[...,1],
                                   cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0
        if hair_mask is not None and index in {0,len(frame_names)//2,len(frame_names)-1}:
            preview=gt.copy();preview[hair_mask]=.65*preview[hair_mask]+.35*np.array([1.,0.,0.])
            cv2.imwrite(str(output/('hair_proxy_'+frame+'.jpg')),
                        np.ascontiguousarray(np.rot90(np.rint(preview[...,::-1]*255).astype(np.uint8))))
        previous=predictions;previous_gt=gt
        write(output/'progress.json',dict(completed=index+1,total=len(frame_names)))
    aggregate={}
    for mode in mode_rows:
        aggregate[mode]={}
        for region in rows[0]['metrics'][mode]:
            values=[r['metrics'][mode][region] for r in rows];count=sum(v['valid_pixels'] for v in values)
            aggregate[mode][region]=dict(valid_pixels=count,
                **{key:sum(v[key]*v['valid_pixels'] for v in values if v['valid_pixels'])/count if count else None
                   for key in ['error_change_mae','prediction_change_mae','gt_change_mae']})
    result=dict(pairs=len(rows),analysis_raster=list(size),aggregate=aggregate,per_pair=rows,
        flow='GT-only bidirectional DIS medium; forward/backward disagreement <=1.5 analysis pixels',
        metric='MAE of motion-warped prediction-error changes; bounded display RGB',
        reference_mask=str(reference) if reference else None,
        reference_mask_sha256=reference_sha,
        hair_mask_is_propagated_fallible_proxy=reference is not None,
        training_or_checkpoint_selection_changed=False,depth_or_hair_topology_ground_truth=False,
        physical_time_stability_proxy=True,capture_timing_calibrated=False)
    write(output/'complete.json',result)
    return result
