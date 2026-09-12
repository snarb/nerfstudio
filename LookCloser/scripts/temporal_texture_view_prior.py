"""Isolated calibration-only near-view source-label prior; no RGB averaging.

This opt-in adapter adds a camera-angle unary preference to the frozen surface
labeler. Projection, visibility, native sampling, camera profiles and geometry
are unchanged. It never modifies the baseline renderer on disk.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import render_smooth_temporal_mesh_video as renderer
from joint_temporal_texture import cameras,read,sha,atomic_json


def angle_weights(rows,target,sigma):
    if not np.isfinite(sigma) or sigma<=0:raise ValueError('Invalid angular prior width')
    source=np.array([r['transform_matrix'] for r in rows])[:,:3,2]
    query=np.array(target['transform_matrix'])[:3,2]
    angles=np.arccos(np.clip(source@query,-1,1))*180/np.pi
    return np.maximum(np.exp(-.5*(angles/sigma)**2),1e-5).astype(np.float32),angles


def install(module):
    original_render=module.render_one;original_select=module.select_surface_sources
    def render(output,record,manifest):
        sigma=read(output/'request.json')['recipe']['target_angle_sigma_degrees']
        rows,_,_=cameras(record['frame_id']);weights,angles=angle_weights(rows,record['camera'],sigma)
        def select(rgb,quality,triangles,**kwargs):
            labels,report=original_select(rgb,quality*weights[:,None],triangles,**kwargs)
            report.update(target_angle_sigma_degrees=sigma,camera_angles_degrees=angles.tolist(),
                          near_view_prior_from_calibration_only=True)
            return labels,report
        module.select_surface_sources=select
        try:return original_render(output,record,manifest)
        finally:module.select_surface_sources=original_select
    module.render_one=render


def initialize(baseline,output,sigma):
    request=deepcopy(renderer.verify_request(baseline))
    request['recipe'].update(target_angle_sigma_degrees=sigma,texture_source_prior='calibration_only_near_view')
    request['script_hashes'][Path(__file__).name]=sha(__file__)
    output.mkdir(parents=True,exist_ok=True);(output/'frames').mkdir(exist_ok=True)
    if (output/'request.json').exists() and read(output/'request.json')!=request:raise ValueError('View-prior request mismatch')
    atomic_json(output/'request.json',request)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,default=renderer.OUTPUT)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--sigma',type=float,default=4.)
    p.add_argument('--frames',nargs='+',default=['000899','000973','001197']);a=p.parse_args()
    renderer.torch.set_num_threads(4);initialize(a.baseline,a.output,a.sigma);install(renderer);renderer.render(a.output,a.frames)


if __name__=='__main__':main()
