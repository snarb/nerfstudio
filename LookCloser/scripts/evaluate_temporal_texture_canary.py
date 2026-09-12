"""Two-time held-out face-only control for temporal texture source priors."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import argparse
import torch
import render_smooth_temporal_mesh_video as renderer
from temporal_texture_view_prior import install
from joint_temporal_texture import ROOT,SOURCE,CALIBRATION,read,sha,atomic_json
from render_patchmatch_camera_path import normalize_frame


def evaluate(baseline,output,prior):
    request=deepcopy(renderer.verify_request(baseline));ids=['000973','001059']
    request['ordered_frame_ids']=ids;request['inventory']=[r for r in request['inventory'] if r['frame_id'] in ids]
    request['source_rows']=[r for r in request['source_rows'] if Path(r['source_dataset']).name in ids]
    request['recipe']['frames']=2;request['purpose']='heldout_face_only_control_not_a_temporal_video'
    cal=read(CALIBRATION);row=next(r for r in cal['frames'] if r['physical_camera']=='F004_B005_1210O9')
    for record in request['inventory']:
        record['camera']=normalize_frame(row,cal,read(record['metadata']));record['camera'].pop('file_path',None)
    request['script_hashes'][Path(__file__).name]=sha(__file__)
    if prior:
        request['recipe']['target_angle_sigma_degrees']=4.
        request['script_hashes']['temporal_texture_view_prior.py']=sha(Path(__file__).with_name('temporal_texture_view_prior.py'))
        install(renderer)
    output.mkdir(parents=True,exist_ok=True);(output/'frames').mkdir(exist_ok=True)
    if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('Canary request mismatch')
    atomic_json(output/'request.json',request);renderer.render(output,ids)
    from score_colmap_patchmatch_tsdf_face import load_display_rgb,load_manual_face_mask,masked_display_metrics,LearnedPerceptualImagePatchSimilarity
    model=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda().eval();results=[]
    with torch.inference_mode():
        for frame in ids:
            gt_path=ROOT/'frames'/frame/'review/F004_B005_1210O9/gt.png';roi=ROOT/'config'/f'face_roi_{frame}.json'
            gt=load_display_rgb(gt_path);mask,_=load_manual_face_mask(roi,gt_path,gt.shape[:2])
            pred=load_display_rgb(output/'frames'/frame/'prediction_native.png')
            metrics=masked_display_metrics(torch.tensor(pred.transpose(2,0,1),device='cuda'),
                torch.tensor(gt.transpose(2,0,1),device='cuda'),torch.tensor(mask,device='cuda'),model)
            results.append({'frame_id':frame,'gt_sha256':sha(gt_path),'roi_sha256':sha(roi),**metrics})
    atomic_json(output/'face_metrics.json',{'rows':results,'heldout_camera':'F004_B005_1210O9','no_full_frame_metrics':True,'prior':prior})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,default=renderer.OUTPUT)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--prior',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);evaluate(a.baseline,a.output,a.prior)


if __name__=='__main__':main()
