"""Locate residual forearm gaps before/after assembly and color-qualified carving.

Read-only input study. Plane intersections are diagnostic probes, not measured
surfaces, metric masks or permission to add geometry.
"""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from joint_temporal_texture import read, sha, atomic_json, project
from study_confidence_depth_prior import unproject, project_integer
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
import study_forearm_plane_transfer_v3 as prior

BASE=Path('/mnt/data/dec5_forearm_color_qualified_curved')
PARENT=Path('/mnt/data/dec5_phase30_early_texture_dynamic_150')
LABELS={0:'outside reference footprint',1:'outside reference skin',2:'reference old mesh hit',
        3:'reference candidate rejected',4:'accepted reference but no initial triangle',
        5:'lost during production transfer/curvature',6:'lost during color-qualified carving',
        7:'covered by final geometry'}
PALETTE=np.array([[50,50,50],[120,120,120],[255,50,40],[240,180,30],
                  [40,200,240],[200,60,230],[70,100,255],[40,200,70]],np.uint8)

def plane_intersections(camera,reference,coefficients):
    yy,xx=np.indices((1080,1920));pose=np.asarray(reference['transform_matrix'])
    center=np.asarray(camera['transform_matrix'])[:3,3];c=np.asarray(coefficients)
    normal=pose[:3,:3]@np.array([c[0]*reference['fl_x']/100,-c[1]*reference['fl_y']/100,
        -(c[0]*reference['cx']/100+c[1]*reference['cy']/100+c[2])])
    direction=unproject(camera,xx.ravel(),yy.ravel(),np.ones(xx.size),offset=.5)-center
    denominator=direction@normal
    with np.errstate(divide='ignore',invalid='ignore'):
        depth=(1-normal@(center-pose[:3,3]))/denominator
    return center+direction*depth[:,None],depth.reshape(1080,1920)

def run(output,frame):
    prior.configure();v1=prior.v2.v1;out=output/frame;out.mkdir(parents=True,exist_ok=True)
    record=next(r for r in read(PARENT/'request.json')['inventory'] if r['frame_id']==frame)
    rows,_,_=v1.cameras(frame);masks=v1.masks(frame)
    reference=next(r for r in rows if r['physical_camera']==v1.NAMES[0])
    root=prior.OUT/frame;analysis=read(root/'analysis.json');ev=np.load(root/'plane/evidence.npz')
    md=np.load(root/'diagnostic.npz')[v1.NAMES[0]+'_mesh']
    paths={'baseline':Path(record['mesh']),'initial':root/'plane/mesh.ply',
           'transferred':BASE/frame/'transferred.ply','guarded':BASE/frame/'guarded.ply'}
    result=read(BASE/frame/'geometry_result.json')
    for name in ['transferred','guarded']:
        assert sha(paths[name])==result['hashes'][name+'.ply']
    request=dict(frame=frame,script_sha256=sha(__file__),mesh_hashes={k:sha(v) for k,v in paths.items()},
        prior_evidence_sha256=sha(root/'plane/evidence.npz'),prior_diagnostic_sha256=sha(root/'diagnostic.npz'),
        diagnostic_plane_only=True,geometry_changed=False,heldout_rgb_used=False)
    if (out/'request.json').exists() and read(out/'request.json')!=request:raise ValueError('Changed diagnostic request')
    atomic_json(out/'request.json',request);scenes={}
    for name,path in paths.items():
        mesh=o3d.io.read_triangle_mesh(str(path));scenes[name]=scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles))
    views={'moving':record['camera'],**{r['physical_camera']:r for r in rows if r['physical_camera'] in masks}}
    summaries=[]
    for name,camera in views.items():
        points,pdepth=plane_intersections(camera,reference,analysis['plane_inverse_coefficients'])
        uv,z=project_integer(reference,points);q=np.rint(np.nan_to_num(uv,nan=-1,posinf=-1,neginf=-1)).astype(int)
        inside=np.isfinite(points).all(1)&(z>0)&(q[:,0]>=0)&(q[:,0]<1920)&(q[:,1]>=0)&(q[:,1]<1080)
        idx=np.flatnonzero(inside);qx,qy=q[idx].T;codes=np.zeros(len(points),np.uint8)
        codes[idx]=1;skin=masks[v1.NAMES[0]][qy,qx];codes[idx[skin]]=3
        old=skin&(md[qy,qx]>0);codes[idx[old]]=2
        codes[idx[ev['accepted'][qy,qx]]]=4
        depths={k:camera_depth(scene,camera)[0] for k,scene in scenes.items()}
        # Separate lost coverage from an unrelated background surface hit.
        covers={k:np.isfinite(d)&(np.abs(d-pdepth)<.012) for k,d in depths.items()}
        flat=codes.copy();codes[(flat==4)&covers['initial'].ravel()]=5
        codes[np.isin(flat,[2,3,4])&covers['transferred'].ravel()]=6
        codes[covers['guarded'].ravel()]=7;codes=codes.reshape(1080,1920)
        if name in masks:roi=masks[name].copy()
        else:
            support=np.zeros(len(points),np.uint8);outside=np.zeros(len(points),bool)
            for row in rows:
                if row['physical_camera'] not in masks:continue
                pix,cz=project(points,[row]);pix=pix[0];cz=cz[0];xy=np.rint(np.nan_to_num(pix,nan=-1,posinf=-1,neginf=-1)).astype(int)
                available=(cz>0)&(pix[:,0]>2)&(pix[:,0]<1917)&(pix[:,1]>2)&(pix[:,1]<1077)
                ids=np.flatnonzero(available);s=masks[row['physical_camera']][xy[ids,1],xy[ids,0]]
                support[ids]+=s;outside[ids]|=~s
            roi=((support>=2)&~outside).reshape(1080,1920)
        missing=roi&~covers['guarded'];original=missing&(codes==2);sel=np.flatnonzero(original.ravel())
        delta=md[q[sel,1],q[sel,0]]-z[sel]
        row=dict(view=name,scope='fixed train skin ROI' if name in masks else 'diagnostic multiview plane envelope, not metric ROI',
            roi_pixels=int(roi.sum()),stage_missing={k:int((roi&~v).sum()) for k,v in covers.items()},
            final_missing_reason={LABELS[k]:int((missing&(codes==k)).sum()) for k in LABELS},
            old_reference_minus_plane_quantiles=np.quantile(delta,[0,.1,.5,.9,1]).tolist() if len(delta) else None)
        summaries.append(row);np.savez_compressed(out/(name+'_diagnostic.npz'),codes=codes,roi=roi,**depths)
        rgbpath=(PARENT/'frames'/frame/'frame.png') if name=='moving' else root/'rgb'/(name+'.png')
        rgb=np.array(Image.open(rgbpath).convert('RGB'))
        if name!='moving':rgb=np.rot90(rgb)
        overlay=rgb.copy();m=np.rot90(missing);overlay[m]=np.rot90(PALETTE[codes])[m]
        panel=Image.new('RGB',(1080,550));draw=ImageDraw.Draw(panel)
        for i,im in enumerate([rgb,overlay]):panel.paste(Image.fromarray(im).crop((0,1400,540,1920)),(i*540,30))
        draw.text((4,5),name+' / RGB',fill='white');draw.text((544,5),'Residual reason (not RGB quality)',fill='white')
        panel.save(out/(name+'_reasons.png'));print(row,flush=True)
    atomic_json(out/'result.json',dict(request_sha256=sha(out/'request.json'),rows=summaries,labels=LABELS,
        artifacts={p.name:sha(p) for p in out.glob('*') if p.is_file() and p.name!='result.json'},
        geometry_changed=False,visual_status='pending'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',required=True,choices=['001029','001033','001037'])
    p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_forearm_residual_topology'))
    p.add_argument('--base',type=Path,default=BASE);a=p.parse_args();BASE=a.base;run(a.output,a.frame)
