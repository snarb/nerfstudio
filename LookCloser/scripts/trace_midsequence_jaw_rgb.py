"""Trace thin dark jaw residuals to actual single-camera linear RGB samples."""
from pathlib import Path
import argparse
import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from joint_temporal_texture import read, sha, atomic_json, project, exr, display, ROOT as COLOR
from render_midsequence_jaw_completion import ROOT
from classify_midsequence_jaw_seam import POLYGON, VIEW
from study_confidence_depth_prior import load_real, project_integer
from review_jaw_repair_transfer import verified_image
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from calibrated_depth_witness import load_images


def run(frame):
    root=ROOT/frame/'seam_diagnosis'; out=root/'source_trace'; out.mkdir(exist_ok=False)
    folder=ROOT/frame/'rgb'/VIEW/'completed'; rgb,r=verified_image(folder,frame)
    q=read(folder/'request.json');entry=q['inventory'][0];data=folder/'frames'/frame
    gt=np.asarray(Image.open(root/'train_gt.png'))
    mask=np.zeros(rgb.shape[:2],np.uint8);cv2.fillPoly(mask,[np.asarray(POLYGON,np.int32)],1)
    luma=rgb.astype(np.float32)@np.array([.2126,.7152,.0722],np.float32)
    target_luma=gt.astype(np.float32)@np.array([.2126,.7152,.0722],np.float32)
    local=cv2.medianBlur(luma,5)
    chosen=mask.astype(bool)&(luma < target_luma-20)&(luma < local-15)
    py,px=np.where(chosen); y,x=px,1919-py
    source_ids=np.asarray(Image.open(data/'source_ids.png'))
    selected=source_ids[y,x]; assert (selected<62).all()
    mesh=o3d.io.read_triangle_mesh(entry['mesh']);v=np.asarray(mesh.vertices,np.float32);t=np.asarray(mesh.triangles)
    depth,faces,bary=camera_depth(scene_for(v,t),entry['camera'])
    f=faces[y,x];b=bary[y,x];weights=np.column_stack([1-b.sum(1),b]);points=(v[t[f]]*weights[:,:,None]).sum(1)
    bq=read(Path('/mnt/data/dec5_jaw_measured_mask_control')/frame/'request.json')
    rows,depths,receipt=load_real(Path(bq['depth_root']),frame);assert receipt==bq['depth_receipt']
    assert [r['physical_camera'] for r in rows]==r['source_cameras']
    uv,z=project(points,rows);rounded=np.rint(uv);uv=np.where(np.abs(uv-rounded)<=.001,rounded,uv)
    pars=np.load(COLOR/'parameters.npz')['log_gain'];gain=np.exp(pars-pars.mean(0,keepdims=True))
    exposure=read(COLOR/'exposure.json')['fixed_exposure_gain']
    source_display,_,rgb_receipt=load_images(frame)
    records=[];examples=[];replayed=np.zeros((len(x),3),np.uint8)
    deltas=np.full(len(x),np.nan);observed_valid=np.zeros(len(x),bool)
    source_boundary=np.zeros(len(x),bool)
    for dy,dx in [(0,-1),(0,1),(-1,0),(1,0)]:source_boundary|=source_ids[y+dy,x+dx]!=selected
    for ci in np.unique(selected):
        j=np.flatnonzero(selected==ci);suv=uv[ci,j];floor=np.floor(suv);frac=suv-floor
        ix,iy=floor[:,0].astype(int),floor[:,1].astype(int);linear=exr(rows[ci]['file_path'])
        sample=np.zeros((len(j),3),np.float32)
        for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
            w=(frac[:,0] if dx else 1-frac[:,0])*(frac[:,1] if dy else 1-frac[:,1])
            sample+=linear[iy+dy,ix+dx]*w[:,None]
        replayed[j]=np.rint(display(np.maximum(sample*gain[ci],0),exposure)*255).clip(0,255).astype(np.uint8)
        native,zq=project_integer(rows[ci],points[j]);xy=np.rint(native).astype(int)
        obs=depths[ci][xy[:,1],xy[:,0]];valid=np.isfinite(obs)&(obs>0)
        observed_valid[j]=valid;deltas[j[valid]]=obs[valid]-zq[valid]
        record=dict(camera=rows[ci]['physical_camera'],pixels=len(j),near_depth=int((valid&(np.abs(obs-zq)<=.0015)).sum()),
            farther_depth=int((valid&(obs-zq>.005)).sum()),unknown_depth=int((~valid).sum()),
            source_boundary_pixels=int(source_boundary[j].sum()))
        # One deterministic median-residual witness for every actually selected camera.
        residual=target_luma[py[j],px[j]]-luma[py[j],px[j]]
        k=int(j[np.argsort(residual)[len(j)//2]]);u,w=np.rint(uv[ci,k]).astype(int)
        items=[(gt,int(px[k]),int(py[k]),'real train target GT'),(rgb,int(px[k]),int(py[k]),'prediction'),
               (source_display[rows[ci]['physical_camera']],u,w,'actual selected source')]
        sheet=Image.new('RGB',(600,240));draw=ImageDraw.Draw(sheet)
        draw.text((3,3),f'{frame} {rows[ci]["physical_camera"]} delta-z={deltas[k]:+.6f}',fill='white')
        for col,(im,cx,cy,name) in enumerate(items):
            patch=Image.fromarray(im).crop((cx-90,cy-90,cx+90,cy+90))
            ImageDraw.Draw(patch).ellipse((87,87,93,93),outline='red',width=1)
            sheet.paste(patch,(col*200,48));draw.text((col*200+3,28),name,fill='white')
        path=out/f'case_{len(examples):02d}.png';sheet.save(path)
        examples.append(dict(camera=rows[ci]['physical_camera'],portrait_xy=[int(px[k]),int(py[k])],
            source_uv=uv[ci,k].tolist(),path=str(path),sha256=sha(path)))
        records.append(record)
    error=np.abs(rgb[py,px].astype(int)-replayed.astype(int))
    assert error.max(initial=0)<=1
    np.savez_compressed(out/'evidence.npz',portrait_xy=np.column_stack([px,py]),points=points,
        selected_source=selected,source_uv=uv,projected_z=z,depth_delta=deltas,measured_valid=observed_valid,
        source_boundary=source_boundary,replayed_rgb=replayed,predicted_rgb=rgb[py,px],gt_rgb=gt[py,px])
    atomic_json(out/'result.json',dict(frame=frame,selected_dark_ridge_pixels=len(x),
        selection_rule='Inside fixed GT jaw polygon; target-minus-predicted luma >20 and 5x5 median-minus-predicted >15',
        selected_for_diagnosis_not_quality_metric=True,records=records,examples=examples,
        max_uint8_rgb_replay_error=int(error.max(initial=0)),depth_receipt=receipt,rgb_receipt=rgb_receipt,
        request_sha256=sha(folder/'request.json'),script_sha256=sha(__file__),
        evidence_sha256=sha(out/'evidence.npz'),geometry_changed=False,renderer_changed=False))
    print(frame,'dark ridge pixels',len(x),'sources',records,'RGB replay max',error.max(initial=0),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',required=True,choices=['001193','001195'])
    run(p.parse_args().frame)
