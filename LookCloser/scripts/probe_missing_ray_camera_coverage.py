"""Bracket missing forearm rays against real camera fields of view.

An extrapolated boundary plane is a diagnostic depth hypothesis, not added mesh
or ground truth. Test a broad depth interval and report frustum counts separately
from actual measured depth agreement.
"""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,cameras,ROOT,exr,display
from study_confidence_depth_prior import robust_fit,unproject,project_integer,support
from import_colmap_mvs_depth_dataset import read_colmap_dense_array


def run():
    base=Path('/mnt/data/dec5_elevated_camera_dynamic_150');control=Path('/mnt/data/dec5_forearm_depth_control_001033')
    row=next(r for r in read(base/'request.json')['inventory'] if r['frame_id']=='001033');camera=row['camera']
    depth=np.rot90(np.load(base/'frames/001033/target_depth.npz')['depth'])
    # Manual adjacent-skin window from actual native target inspection.
    yy,xx=np.indices(depth.shape);region=(xx>=320)&(xx<430)&(yy>=1670)&(yy<1750)&(depth>0)
    cy,cx=np.nonzero(region);z=depth[region];design=np.column_stack((cx/1920,cy/1920,np.ones(len(cx))))
    coef,rmse=robust_fit(design,1/z)
    queries=[(365,1810),(365,1850),(365,1890)]
    rows,_,meta=cameras('001033');scale=read(meta)['dataparser_scale'];spec=read(control/'staged63/transforms.json')
    lookup={r['physical_camera']:r for r in spec['frames']};depths=[]
    for r in rows:
        p=control/'pipeline/dense/stereo/depth_maps'/(lookup[r['physical_camera']]['file_path']+'.geometric.bin')
        depths.append(read_colmap_dense_array(p)[...,0]*scale)
    records=[];native=np.asarray(Image.open(base/'frames/001033/frame.png'));preview=Image.fromarray(native);draw=ImageDraw.Draw(preview)
    for px,py in queries:
        estimate=1/(np.array([px/1920,py/1920,1])@coef)
        samples=np.linspace(estimate-.02,estimate+.02,101)
        # Renderer rays use pixel centers; integer projector expects +0.5 here.
        points=unproject(camera,np.full(101,1919-py),np.full(101,px),samples,offset=.5)
        fov=np.zeros(101,np.uint8);projections=[]
        for r in rows:
            uv,d=project_integer(r,points)
            valid=(d>0)&(uv[:,0]>=0)&(uv[:,0]<1920)&(uv[:,1]>=0)&(uv[:,1]<1080)
            fov+=valid
            projections.append(dict(camera=r['physical_camera'],inside_at_estimate=bool(valid[50]),
                native_xy_at_estimate=uv[50].tolist(),any_inside_interval=bool(valid.any())))
        votes,free=support(points,camera,rows,depths)
        records.append(dict(portrait_xy=[px,py],saved_mesh_hit=bool(depth[py,px]>0),estimated_z=float(estimate),
            depth_bracket=[float(samples[0]),float(samples[-1])],
            frustum_at_estimate=int(fov[50]),frustum_min=int(fov.min()),frustum_max=int(fov.max()),
            depth_agreement_at_estimate=int(votes[50]),depth_agreement_max=int(votes.max()),
            projections=projections))
        draw.ellipse((px-5,py-5,px+5,py+5),outline='yellow',width=2)
    out=control/'missing_ray_coverage';out.mkdir(exist_ok=True)
    preview.crop((180,1580,530,1920)).save(out/'queries.png')
    inside={r['camera']:r for r in records[1]['projections'] if r['inside_at_estimate']}
    center=np.asarray(camera['transform_matrix'])[:3,3]
    selected=sorted([r for r in rows if r['physical_camera'] in inside],
        key=lambda r:np.linalg.norm(np.asarray(r['transform_matrix'])[:3,3]-center))[:6]
    profiles=read(ROOT/'camera_profiles.json');gains=dict(zip(profiles['physical_cameras'],profiles['rgb_gain']))
    exposure=read(ROOT/'exposure.json')['fixed_exposure_gain'];panel=Image.new('RGB',(960,688));draw=ImageDraw.Draw(panel);images=[]
    for j,r in enumerate(selected):
        u,v=inside[r['physical_camera']]['native_xy_at_estimate'];px,py=int(round(v)),int(round(1919-u))
        rgb=np.rint(display(exr(r['file_path'])*gains[r['physical_camera']],exposure)*255).clip(0,255).astype(np.uint8)
        im=Image.fromarray(np.rot90(rgb));d=ImageDraw.Draw(im);d.ellipse((px-5,py-5,px+5,py+5),outline='yellow',width=2)
        box=[px-160,py-160,px+160,py+160];im=im.crop(box);im.save(out/f'reference_{j}.png')
        x,y=j%3*320,j//3*344;panel.paste(im,(x,y+24));draw.text((x+3,y+4),r['physical_camera'],fill='white')
        iu,iv=int(round(u)),int(round(v));stage_stats={}
        for kind in ['photometric','geometric']:
            path=control/'pipeline/dense/stereo/depth_maps'/(lookup[r['physical_camera']]['file_path']+'.'+kind+'.bin')
            d=read_colmap_dense_array(path)[...,0]*scale;patch=d[max(0,iv-5):iv+6,max(0,iu-5):iu+6];valid=patch>0
            stage_stats[kind]=dict(valid_fraction=float(valid.mean()),median_depth=float(np.median(patch[valid])) if valid.any() else None,source_sha256=sha(path))
        images.append(dict(physical_camera=r['physical_camera'],source_sha256=sha(r['file_path']),crop=box,depth_11x11=stage_stats))
    panel.save(out/'inside_camera_references.png')
    atomic_json(out/'result.json',dict(frame='001033',script_sha256=sha(__file__),
        parent_request_sha256=sha(base/'request.json'),control_request_sha256=sha(control/'request.json'),
        hypothesis_only=True,plane_inverse_depth_rmse=rmse,plane_sample_count=len(z),
        no_geometry_modified=True,queries=records,real_train_reference_images=images))
    print([{k:v for k,v in r.items() if k!='projections'} for r in records],flush=True)


if __name__=='__main__':run()
