"""Native RGB evidence for three spatially separated weak crown triangles.

This is a diagnostic selection, not a geometry-edit mask or deletion decision.
Preserve all62 per-camera observations and inspect a deterministic six-view subset.
"""
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,cameras,project
from diagnose_original_crown_support import ROOT,FRAMES,MASKS
from calibrated_depth_witness import load_images
from study_confidence_depth_prior import load_real,project_integer
from transfer_close_boundary_completion import SOURCE


def run():
    all_records=[]
    for frame in FRAMES:
        folder=ROOT/frame;result=read(folder/'result.json');request=read(folder/'request.json')
        assert result['request_sha256']==sha(folder/'request.json')
        assert sha(folder/'evidence.npz')==result['hashes']['evidence.npz']
        a=np.load(folder/'evidence.npz');mesh=o3d.io.read_triangle_mesh(request['source_mesh'])
        v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles);rows,_,_=cameras(frame)
        base=read(SOURCE/frame/'request.json');rows,depths,receipt=load_real(Path(base['depth_root']),frame)
        assert receipt==request['depth_receipt']
        images,_,rgb_receipt=load_images(frame);assert rgb_receipt==request['rgb_receipt']
        masks=np.load(MASKS/frame/'masks.npz')['masks'];names=read(MASKS/frame/'cameras.json')
        available=np.flatnonzero(a['both']&np.isin(a['query_triangles'],a['crown_triangles']))
        order=available[np.argsort(-a['refined_mask_outside'][available],kind='stable')]
        chosen=[];centers=[]
        for index in order:
            c=v[t[a['query_triangles'][index]]].mean(0)
            if all(np.linalg.norm(c-x)>=.002 for x in centers):chosen.append(int(index));centers.append(c)
            if len(chosen)==3:break
        assert len(chosen)==3
        records=[]
        for number,index in enumerate(chosen):
            triangle=int(a['query_triangles'][index]);pts=v[t[triangle]];center=pts.mean(0)
            observations=[]
            for ci,(row,depth) in enumerate(zip(rows,depths)):
                uv,z=project(np.concatenate([pts,center[None]]),[row]);uv=uv[0];z=z[0]
                xy=np.rint(uv).astype(int);valid=(z>0)&(xy[:,0]>=4)&(xy[:,0]<1916)&(xy[:,1]>=4)&(xy[:,1]<1076)
                if not valid.all():continue
                mask=masks[names.index(row['physical_camera'])];inside=mask[xy[:,1],xy[:,0]]
                clear=[not mask[y-4:y+5,x-4:x+5].any() for x,y in xy]
                depthuv,dz=project_integer(row,center[None]);dx,dy=np.rint(depthuv[0]).astype(int)
                observed=float(depth[dy,dx]);observed_valid=np.isfinite(observed) and observed>0
                observations.append(dict(camera_index=ci,camera=row['physical_camera'],uv=uv.tolist(),
                    mask_inside=inside.tolist(),all_four_clear_background=bool(all(clear)),
                    center_observed_depth=observed if observed_valid else None,center_mesh_depth=float(dz[0]),
                    center_depth_error=float(abs(observed-dz[0])) if observed_valid else None))
            native=next(x for x in observations if x['camera']==request['native_camera']['physical_camera'])
            selected=[native]
            # Two strong background conflicts, then nearest measured agreement,
            # then remaining observations to expose contradictory evidence too.
            clear=[x for x in observations if x['all_four_clear_background']]
            inside=[x for x in observations if all(x['mask_inside'])]
            nearest=sorted(observations,key=lambda x:x['center_depth_error'] if x['center_depth_error'] is not None else float('inf'))
            for pool,limit in [(clear,3),(nearest,4),(inside,6),(observations,6)]:
                for item in pool:
                    if item not in selected:selected.append(item)
                    if len(selected)>=limit:break
                if len(selected)>=6:break
            selected=selected[:6]
            canvas=Image.new('RGB',(3*320,2*355),(20,20,20));draw=ImageDraw.Draw(canvas)
            for j,obs in enumerate(selected):
                rgb=images[obs['camera']];uv=np.asarray(obs['uv']);cx,cy=np.rint(uv[-1]).astype(int)
                crop=Image.fromarray(rgb).crop((cx-80,cy-80,cx+80,cy+80)).resize((320,320),Image.Resampling.NEAREST)
                local=(uv[:3]-[cx-80,cy-80])*2
                cd=ImageDraw.Draw(crop);cd.line([tuple(x) for x in np.vstack([local,local[:1]])],fill=(255,40,40),width=2)
                cd.ellipse((157,157,163,163),outline=(0,255,255),width=1)
                x=(j%3)*320;y=(j//3)*355;canvas.paste(crop,(x,y+35))
                error=obs['center_depth_error'];label='missing' if error is None else f'{error:.5f}'
                draw.text((x+3,y+2),obs['camera'],fill='white')
                draw.text((x+3,y+17),f'clearBG={obs["all_four_clear_background"]} |dz|={label}',fill='white')
            path=folder/f'projection_{number:02d}.png';canvas.save(path)
            records.append(dict(triangle=triangle,sample_depth_votes=a['depth_votes'][index].tolist(),
                refined_mask_vetoes=int(a['refined_mask_outside'][index]),
                clear_background_cameras=sum(x['all_four_clear_background'] for x in observations),
                native_camera=request['native_camera']['physical_camera'],observations=observations,
                displayed_cameras=[x['camera'] for x in selected],panel=str(path),panel_sha256=sha(path)))
        atomic_json(folder/'projection_review.json',dict(records=records,selection_not_deletion_rule=True,
            script_sha256=sha(__file__),visual_status='pending',geometry_changed=False))
        all_records.append(dict(frame=frame,selected=[dict(triangle=r['triangle'],votes=r['sample_depth_votes'],
            outside=r['refined_mask_vetoes'],clear_background_cameras=r['clear_background_cameras']) for r in records]))
        print(all_records[-1],flush=True)
    atomic_json(ROOT/'projection_summary.json',dict(records=all_records,geometry_changed=False))


if __name__=='__main__':run()
