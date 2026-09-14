"""Measured and color-qualified evidence in the jaw veto camera, not a mask fix."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import numpy as np
from PIL import Image,ImageDraw
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json,ROOT,exr,display
from study_confidence_depth_prior import load_real,unproject,project_integer,support
from forearm_rgb_witnesses import color_errors
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth

BASE=Path('/mnt/data/dec5_jaw_repair_transfer')
OUT=Path('/mnt/data/dec5_jaw_veto_measured_evidence')


def run(output,frame):
    request=read(BASE/frame/'request.json'); rows,depths,receipt=load_real(Path(request['depth_root']),frame)
    if receipt!=request['depth_receipt']:raise ValueError('Changed measured depths')
    mask_details=read(BASE/'mask_disagreement'/frame/'result.json')
    folder=output/frame;folder.mkdir(parents=True,exist_ok=True)
    record=dict(frame=frame,parent_request_sha256=sha(BASE/frame/'request.json'),
        mask_details_sha256=sha(BASE/'mask_disagreement'/frame/'result.json'),depth_receipt=receipt,
        scripts={n:sha(Path(__file__).with_name(n)) for n in [Path(__file__).name,'forearm_rgb_witnesses.py',
            'diagnose_forearm_color_witnesses.py','study_confidence_depth_prior.py']},
        profiles_sha256=sha(ROOT/'parameters.npz'),exposure_sha256=sha(ROOT/'exposure.json'),
        protocol='query-observed depth; other-view roundtrip/parallax; 5x5 calibrated RGB .12/chroma .04',
        neighborhood_halfwidth=24,geometry_changed=False,mask_changed=False,heldout_used=False)
    if (folder/'request.json').exists() and read(folder/'request.json')!=record:raise ValueError('Frozen evidence mismatch')
    atomic_json(folder/'request.json',record)
    if not mask_details['details']:
        atomic_json(folder/'result.json',dict(request_sha256=sha(folder/'request.json'),records=[],reason='No mask-vetoed proposals in fixed diagnostic ROI'))
        print(frame,'no local mask disagreement',flush=True);return
    log_gain=np.load(ROOT/'parameters.npz')['log_gain'];gain=read(ROOT/'exposure.json')['fixed_exposure_gain']
    def load(pair):
        i,row=pair
        return row['physical_camera'],np.rint(display(exr(row['file_path'])*np.exp(log_gain[i]),gain)*255).clip(0,255).astype(np.uint8)
    with ThreadPoolExecutor(max_workers=4) as pool:images=dict(pool.map(load,enumerate(rows)))
    old=o3d.io.read_triangle_mesh(request['source_mesh']);v,t=np.asarray(old.vertices),np.asarray(old.triangles)
    ev=np.load(BASE/frame/'evidence.npz');raw=np.concatenate([t,ev['proposals']]);oldscene=scene_for(v,t);newscene=scene_for(v,raw)
    results=[]
    for detail in mask_details['details']:
        name=detail['camera'];index=next(i for i,r in enumerate(rows) if r['physical_camera']==name);camera=rows[index]
        xy=np.unique(np.rint(detail['rejected_xy']).astype(int),axis=0)
        x,y=np.rint(np.median(xy,axis=0)).astype(int);yy,xx=np.mgrid[y-24:y+25,x-24:x+25]
        observed=depths[index][yy,xx];available=np.isfinite(observed)&(observed>0)
        points=unproject(camera,xx[available],yy[available],observed[available])
        chroma,rgb=color_errors(points,camera,rows,depths,images)
        geo=np.isfinite(chroma).sum(0);qualified=((chroma<=.04)&(rgb<=.12)).sum(0)
        counts,_=support(points,camera,rows,depths)
        if not np.array_equal(counts,geo):raise ValueError('Inconsistent geometric witnesses')
        actual=dict(camera,cx=camera['cx']+.5,cy=camera['cy']+.5)
        od,_,_=camera_depth(oldscene,actual);nd,nids,_=camera_depth(newscene,actual)
        support_map=np.zeros(observed.shape,int);support_map[available]=qualified
        old_local,new_local=od[yy,xx],nd[yy,xx]
        near=available&np.isfinite(new_local)&(np.abs(observed-new_local)<=.001)
        far=available&np.isfinite(new_local)&(observed>new_local+.003)
        raw_visible=(nids[yy,xx]>=len(t))&(nids[yy,xx]<len(raw))
        records=[]
        for qx,qy in xy:
            j,i=qy-(y-24),qx-(x-24)
            records.append(dict(x=int(qx),y=int(qy),observed_depth=float(observed[j,i]),available=bool(available[j,i]),
                qualified_other_views=int(support_map[j,i]),agrees_with_raw_ray=bool(near[j,i]),
                observed_farther=bool(far[j,i]),raw_added_face_visible=bool(raw_visible[j,i])))
        colors=np.zeros((*observed.shape,3),np.uint8);colors[available]=[90,90,90]
        colors[near&(support_map>=3)]=[0,220,120];colors[far&(support_map>=3)]=[240,60,50]
        # Tiny map is enlarged with nearest-neighbor only for reading sample locations.
        panel=Image.new('RGB',(654,318));draw=ImageDraw.Draw(panel)
        context=Image.fromarray(images[name]).crop((x-90,y-90,x+90,y+90)).rotate(90)
        panel.paste(context,(0,24));panel.paste(Image.fromarray(np.rot90(colors)).resize((294,294),Image.Resampling.NEAREST),(180,24))
        draw.text((3,3),name+' RGB / supported near=green, far=red',fill='white')
        draw.text((480,40),'grey: observed',fill='white');draw.text((480,62),'black: missing',fill='white')
        path=folder/(name+'.png');panel.save(path)
        np.savez_compressed(folder/(name+'.npz'),xy=np.column_stack([xx.ravel(),yy.ravel()]),observed=observed,
            available=available,qualified_views=support_map,near_raw=near,far_raw=far,raw_added_visible=raw_visible,
            chroma=chroma,rgb=rgb,points=points)
        results.append(dict(camera=name,unique_rejected_integer_pixels=records,
            neighborhood_observed=int(available.sum()),qualified_observed=int((support_map>=3).sum()),
            qualified_near_raw=int((near&(support_map>=3)).sum()),qualified_far_raw=int((far&(support_map>=3)).sum()),
            new_faces_visible=int(raw_visible.sum()),panel_sha256=sha(path),arrays_sha256=sha(folder/(name+'.npz'))))
    atomic_json(folder/'result.json',dict(request_sha256=sha(folder/'request.json'),records=results,
        source_rgb_sha256={r['file_path']:sha(r['file_path']) for r in rows},geometry_changed=False,mask_changed=False))
    print(frame,results,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--frame',choices=['001193','001195'],required=True);a=p.parse_args();run(a.output,a.frame)
