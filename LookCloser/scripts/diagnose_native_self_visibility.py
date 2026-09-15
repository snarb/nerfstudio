"""Check why a raycast point is rejected by the identical native source camera."""
from pathlib import Path
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json,cameras,project
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth
from native_texture_footprint import sample_native,snap_centers,relevant_tap

ROOT=Path('/mnt/data/dec5_native_self_visibility')


def run():
    ROOT.mkdir(exist_ok=False);torch.set_num_threads(2);records=[]
    for frame,name in [('001193','G004_C005_121037'),('001123','K004_C005_1210BC')]:
        base=Path('/mnt/data/dec5_central_train_pose_transfer')/frame/('native_'+name)
        request=read(base/'request.json');e=request['inventory'][0];rows,_,_=cameras(frame);row=next(r for r in rows if r['physical_camera']==name)
        np.testing.assert_array_equal(row['transform_matrix'],e['camera']['transform_matrix'])
        m=o3d.io.read_triangle_mesh(e['mesh']);v=np.asarray(m.vertices,np.float32);t=np.asarray(m.triangles,np.uint32)
        d,ids,b=camera_depth(scene_for(v,t),row);hit=np.isfinite(d);yy,xx=np.nonzero(hit)
        point=(v[t[ids[hit]]]*np.column_stack((1-b[hit].sum(1),b[hit]))[:,:,None]).sum(1)
        uv,z=project(point,[row]);error=uv[0]-np.column_stack((xx,yy));q=snap_centers(torch.tensor(uv[:,None],device='cuda'));zq=torch.tensor(z,device='cuda')
        masksroot=Path(e['source_masks']['root']);masknames=read(masksroot/'cameras.json');mask=np.load(masksroot/'masks.npz')['masks'][masknames.index(name)]
        depth=torch.tensor(np.where(hit&(mask>0),d,0)[None,None],device='cuda')
        value=sample_native(depth,q)[:,0,0];valid=(zq>0)&(value>0)&((value-zq).abs()<.0015*zq)
        stages={'source_mask_zero':mask[hit]==0,'bilinear_depth_reject':~valid.cpu().numpy()[0]}
        for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
            tap=sample_native(depth,q.floor()+q.new_tensor([dx,dy]))[:,0,0]
            valid&=((tap>0)&((tap-zq).abs()<.003*zq))|~relevant_tap(q,dx,dy)
        stages['four_tap_reject']=~valid.cpu().numpy()[0]
        selected=np.array(Image.open(Path('/mnt/data/dec5_pixel_angular_head_texture')/frame/('native_'+name)/'frames'/frame/'source_ids.png'))
        idx=next(i for i,r in enumerate(rows) if r['physical_camera']==name)
        stages['different_selected_source']=selected[hit]!=idx
        # The fixed boxes include background; counts are diagnostic, not metrics.
        px=yy;py=1919-xx;box=(700,830,810,1030) if frame=='001193' else (450,1010,820,1200)
        x0,y0,x1,y1=box;roi=(px>=x0)&(px<x1)&(py>=y0)&(py<y1)
        stats={k:dict(all=int(a.sum()),in_box=int(a[roi].sum())) for k,a in stages.items()}
        stats['projection_error_max_per_axis']=np.max(np.abs(error),0).tolist()
        stats['projection_error_abs_quantiles']=np.quantile(np.max(np.abs(error),1),[.5,.9,.99,1]).tolist()
        stats['near_pixel_center_within_001']=int((np.max(np.abs(error),1)<=.001).sum())
        stats['visible_points']=int(hit.sum())
        np.savez_compressed(ROOT/(frame+'.npz'),pixel_xy=np.column_stack((xx,yy)),error=error,roi=roi,**stages)
        records.append(dict(frame=frame,physical_camera=name,stats=stats,roi_box=box,request_sha256=sha(base/'request.json')))
    atomic_json(ROOT/'result.json',dict(records=records,script_sha256=sha(__file__),no_quality_metrics=True))
    print(records,flush=True)


if __name__=='__main__':run()
