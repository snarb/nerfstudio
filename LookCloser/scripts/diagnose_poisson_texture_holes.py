"""Read-only replay of source visibility for black RGB pixels with valid geometry."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import torch
import open3d as o3d
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json,cameras,project,sample
from study_poisson_jaw_completion import OUT,FRAME
from diffusion_mesh_repair import scene_for
from bake_joint_temporal_mesh import camera_depth


def run():
    torch.set_num_threads(4)
    root=OUT/'interpolated/rgb'/FRAME/'F004_E005_1210FP/repaired';folder=root/'frames'/FRAME
    request=read(root/'request.json');record=request['inventory'][0]
    complete=read(folder/'complete.json');assert complete['request_sha256']==sha(root/'request.json')
    for p,h in complete['hashes'].items():assert sha(folder/p)==h
    depth=np.rot90(np.load(folder/'target_depth.npz')['depth']);rgb=np.array(Image.open(folder/'frame.png'))
    polygon=[[610,1090],[655,1120],[693,1134],[710,1146],[719,1152],[711,1215],[705,1230],[690,1200],[668,1175],[635,1145],[610,1125]]
    mask=np.zeros(depth.shape,np.uint8);cv2.fillPoly(mask,[np.array(polygon,np.int32)],1)
    y,x=np.where(mask.astype(bool)&(depth>0)&(rgb.max(2)==0));native_x=1919-y;native_y=x
    mesh=o3d.io.read_triangle_mesh(record['mesh']);v=np.asarray(mesh.vertices,np.float32);t=np.asarray(mesh.triangles,np.uint32);scene=scene_for(v,t)
    d,ids,b=camera_depth(scene,record['camera']);faces=ids[native_y,native_x];weights=b[native_y,native_x]
    weights=np.column_stack([1-weights.sum(1),weights]);points=(v[t[faces]]*weights[:,:,None]).sum(1)
    rows,_,_=cameras(FRAME);ms=record['source_masks'];mr=Path(ms['root']);assert sha(mr/'masks.npz')==ms['masks_sha256']
    masks=dict(zip(read(mr/'cameras.json'),np.load(mr/'masks.npz')['masks']))
    def raycast(row):
        z=camera_depth(scene,row)[0];valid=np.isfinite(z)&masks[row['physical_camera']].astype(bool)
        return np.where(valid,z,0)
    with ThreadPoolExecutor(max_workers=4) as pool:ds=np.stack(list(pool.map(raycast,rows)))
    source_depth=torch.tensor(ds[:,None]);uv,z=project(points,rows);q=torch.tensor(uv[:,None]);zt=torch.tensor(z)
    measured=sample(source_depth,q)[:,0,0]
    raw=(zt>0)&(measured>0)&((measured-zt).abs()<.0015*zt)
    raw&=(q[:,0,:,0]>2)&(q[:,0,:,0]<1917)&(q[:,0,:,1]>2)&(q[:,0,:,1]<1077)
    strict=raw.clone();weighted=raw.clone();tap_results=[]
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        tap=sample(source_depth,q.floor()+q.new_tensor([dx,dy]))[:,0,0]
        good=(tap>0)&((tap-zt).abs()<.003*zt);fraction=(q-q.floor())[:,0]
        weight=(fraction[:,:,0] if dx else 1-fraction[:,:,0])*(fraction[:,:,1] if dy else 1-fraction[:,:,1])
        strict&=good;weighted&=good|(weight<=.001)
        tap_results.append(dict(dx=dx,dy=dy,bad_camera_pixel_pairs=int((raw&~good).sum()),
            bad_with_weight_le_001=int((raw&~good&(weight<=.001)).sum())))
    out=OUT/'texture_diagnosis';out.mkdir(exist_ok=False)
    np.savez_compressed(out/'evidence.npz',portrait_xy=np.column_stack([x,y]),points=points,raw=raw.numpy(),strict=strict.numpy(),
        hypothetical_weighted=weighted.numpy(),uv=uv,z=z)
    result=dict(script_sha256=sha(__file__),render_receipt_sha256=sha(folder/'complete.json'),
        black_rgb_with_depth=len(x),source_camera_names=[r['physical_camera'] for r in rows],
        raw_visible_sources_per_pixel=raw.sum(0).tolist(),strict_sources_per_pixel=strict.sum(0).tolist(),
        hypothetical_weighted_sources_per_pixel=weighted.sum(0).tolist(),tap_summary=tap_results,
        hypothetical_tap_weight_threshold=.001,renderer_changed=False,RGB_generated=False,heldout_used=False,
        evidence_sha256=sha(out/'evidence.npz'))
    atomic_json(out/'result.json',result);print(result,flush=True)


if __name__=='__main__':run()
