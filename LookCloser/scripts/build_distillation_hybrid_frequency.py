"""Initialize the Frequency Grid from real-train frequencies and trusted mesh geometry.

Each original-surface point collects visible real observations. Their median
projected frequency is quantized before voxel max aggregation, following the
local paper's initialization. Unknown points/voxels retain the conservative
runtime fallback; synthetic RGB and the held-out image never supply frequencies.
"""
import argparse
import gzip
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
import trimesh
from mesh_distillation_background import BUNDLE,read,write,sha,check
from prepare_distillation_actor_probe import link
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.pipelines.mesh_distillation_pipeline import ObservedFrequencyGrid
from nerfstudio.model_components.mesh_distillation import projected_frequency


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--points',type=int,default=200000);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'frequency_grid.pt').exists():raise ValueError('Use a new hybrid initialization output')
    torch.set_num_threads(2);started=time.monotonic();rng=np.random.default_rng(42)
    meta=read(a.source/'transforms.json');train=set(meta['train_filenames'])
    synth=read(BUNDLE/'synthetic/transforms.json');teacher={r['physical_camera']:r for r in synth['frames'] if 'physical_camera' in r}
    mesh=trimesh.load(BUNDLE/'mesh/teacher.ply',process=False)
    original=np.load(BUNDLE/'mesh/face_matches_original_tsdf.npy').astype(bool)
    if original.shape!=(len(mesh.faces),):raise ValueError('Original-face provenance mismatch')
    triangles=np.asarray(mesh.triangles)[original];areas=np.asarray(mesh.area_faces)[original]
    chosen=triangles[rng.choice(len(triangles),a.points,p=areas/areas.sum())]
    uv=rng.random((a.points,2));uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1]
    positions=chosen[:,0]+uv[:,:1]*(chosen[:,1]-chosen[:,0])+uv[:,1:]*(chosen[:,2]-chosen[:,0])
    positions=torch.tensor(positions,device='cuda',dtype=torch.float32)
    aabb=torch.tensor(meta['distillation']['actor_bounds'],device='cuda');frequencies=[];camera_receipts=[]
    for row in meta['frames']:
        for key in ['file_path','mask_path','empty_mask_path','evaluation_mask_path']:
            if key in row:link(a.source/row[key],a.output/row[key])
        if row['file_path'] not in train:
            # Nerfstudio requires a depth filename for every frame once any is
            # present. Evaluation gets an explicitly unknown (zero) depth map.
            row['depth_file_path']=f"unknown_depth/{Path(row['file_path']).stem}.npy.gz"
            target=a.output/row['depth_file_path'];target.parent.mkdir(exist_ok=True)
            with gzip.open(target,'wb') as f:np.save(f,np.zeros((row['h'],row['w']),dtype=np.float32))
            continue
        t=teacher[row['physical_camera']]
        for key in ['transform_matrix','fl_x','fl_y','cx','cy','w','h']:
            if not np.allclose(row[key],t[key],atol=1e-9,rtol=0):raise ValueError('Real/teacher camera gauge mismatch')
        row['depth_file_path']=f"teacher_depth/{Path(row['file_path']).stem}.npy.gz"
        link(BUNDLE/'synthetic'/t['depth_file_path'],a.output/row['depth_file_path'])
        with gzip.open(a.output/row['depth_file_path'],'rb') as f:depth=torch.from_numpy(np.load(f)).cuda()
        stem=Path(row['file_path']).stem;folder=a.source/'lookcloser_frequencies'
        receipt=read(folder/f'{stem}.receipt.json')
        if receipt['request']['rgb_sha256']!=sha(a.source/row['file_path']) or receipt['frequency_sha256']!=sha(folder/f'{stem}.pt'):
            raise ValueError('Real frequency source identity mismatch')
        freq=torch.load(folder/f'{stem}.pt',map_location='cuda',weights_only=True)
        valid=torch.load(folder/f'{stem}.valid.pt',map_location='cuda',weights_only=True)
        pose=torch.tensor(row['transform_matrix'],device='cuda');q=(positions-pose[:3,3])@pose[:3,:3];z=-q[:,2]
        x=row['fl_x']*q[:,0]/z+row['cx'];y=-row['fl_y']*q[:,1]/z+row['cy']
        xi=x.floor().long().clamp(0,row['w']-1);yi=y.floor().long().clamp(0,row['h']-1)
        observed=depth[yi,xi];visible=(z>0)&(x>=0)&(x<row['w'])&(y>=0)&(y<row['h'])
        visible &= (observed>0)&torch.isfinite(observed)&((z-observed).abs()<.0015)&valid[yi//8,xi//8]
        intrinsics=[torch.tensor(row[key],device='cuda') for key in ['fl_x','fl_y','w','h']]
        projected=projected_frequency(freq[yi//8,xi//8],*intrinsics,z,(aabb[1]-aabb[0]))
        frequencies.append(torch.where(visible,projected,float('nan')))
        camera_receipts.append(dict(camera=row['physical_camera'],visible_points=int(visible.sum()),
            frequency_sha256=sha(folder/f'{stem}.pt'),trusted_depth_sha256=sha(a.output/row['depth_file_path'])))
    values=torch.stack(frequencies);count=torch.isfinite(values).sum(0);median=values.nanmedian(0).values
    grid=ObservedFrequencyGrid(SceneBox(aabb),resolution=128,num_levels=16,min_res=16.,max_res=8192.).cuda();trusted=(count>=2)&torch.isfinite(median)
    grid.update_max(positions[trusted],grid.freq_to_level(median[trusted]));grid.initialized.copy_(grid.observed.any())
    torch.save(grid.state_dict(),a.output/'frequency_grid.pt')
    link(a.source/'lookcloser_frequencies',a.output/'lookcloser_frequencies')
    meta['distillation']['frequency_initialization']='Median visible real-train frequency per trusted original-mesh point, then voxel max; >=2 views'
    meta['distillation']['depth_supervision']='Gated teacher depths at identical real train poses; missing teacher depth does not mask real RGB'
    write(a.output/'transforms.json',meta)
    write(a.output/'initialization.json',dict(points=a.points,observed_points=int(trusted.sum()),observed_voxels=int(grid.observed.sum()),
        grid_histogram=torch.bincount(grid.grid[grid.observed].long(),minlength=16).tolist(),views=camera_receipts,
        actual_eval_used=False,synthetic_rgb_used=False,grid_resolution=128,min_res=16,max_res=8192,
        median_policy='torch.nanmedian; lower middle for even observation counts',
        mesh_sha256=sha(BUNDLE/'mesh/teacher.ply'),original_faces_sha256=sha(BUNDLE/'mesh/face_matches_original_tsdf.npy'),
        script_sha256=sha(__file__),frequency_grid_sha256=sha(a.output/'frequency_grid.pt')))
    check(a.output,'hybrid_frequency_initialization',len(frequencies),started,finished=True)


if __name__=='__main__':main()
