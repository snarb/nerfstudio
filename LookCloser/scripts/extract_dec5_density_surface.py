#!/usr/bin/env python3
"""Extract one canonical LookCloser isodensity surface without fusing view depths.

The single isovalue is the median density at qualified fitting anchors. No
validation/test RGB or threshold sweep is used. Transparent wisps need fallback.
"""
from pathlib import Path
import argparse,json
import numpy as np
import torch,yaml
from dec5_hair import unproject
from export_dec5_hair_volume import fingerprint
from run_dec5_hair_campaign import atomic_json,sha


def extract(data,training,checkpoint,anchors,output,resolution):
    import open3d as o3d
    from skimage.measure import marching_cubes
    if resolution<64 or resolution>768:raise ValueError('Screen resolution must be 64..768')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(12*(1<<30)/torch.cuda.get_device_properties(0).total_memory)
    request=json.loads((training/'request.json').read_text());receipt=json.loads(anchors.with_suffix('.json').read_text())
    if (request['data_sha256']!=sha(data/'transforms.json') or receipt['dataset_sha256']!=request['data_sha256']
            or receipt['npz_sha256']!=sha(anchors) or receipt.get('uses_validation_rgb') is not False
            or receipt.get('uses_test_rgb') is not False):raise ValueError('Frozen train-only geometry inputs required')
    cfg=yaml.load(next(training.rglob('config.yml')).read_text(),Loader=yaml.Loader)
    cfg.pipeline.datamanager.data=data;cfg.pipeline.datamanager.dataparser.data=data
    pipe=cfg.pipeline.setup(device='cuda',test_mode='val');state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    pipe.load_state_dict(state['pipeline'],strict=True);model=pipe.model;model.current_train_step=state['step'];pipe.eval();before=fingerprint(model)
    payload=json.loads((data/'transforms.json').read_text());by={Path(f['file_path']).name:f for f in payload['frames']}
    dataset=pipe.datamanager.train_dataset;frames=[by[Path(p).name] for p in dataset.image_filenames];names={f['physical_camera']:f for f in frames}
    raw=np.asarray([f['transform_matrix'] for f in frames]);normalized=dataset.cameras.camera_to_worlds.cpu().numpy()
    rotation=normalized[0,:3,:3]@raw[0,:3,:3].T
    scale=np.linalg.norm(normalized[1,:3,3]-normalized[0,:3,3])/np.linalg.norm(raw[1,:3,3]-raw[0,:3,3]);translation=normalized[0,:3,3]-scale*rotation@raw[0,:3,3]
    if np.linalg.norm(scale*raw[:,:3,3]@rotation.T+translation-normalized[:,:3,3],axis=-1).max()>1e-4:raise ValueError('Camera similarity mismatch')
    a=np.load(anchors);xyz=np.empty((len(a['uv']),3),np.float32)
    for index,name in enumerate(a['camera_names']):
        if name not in names:raise ValueError('Anchor camera is not training')
        mask=a['camera']==index;xyz[mask]=unproject(a['uv'][mask],a['camera_z'][mask],names[str(name)])
    fit=xyz[~a['validation'].astype(bool)]
    def query(points):
        world=torch.as_tensor(scale*points@rotation.T+translation,device='cuda',dtype=torch.float32)
        with torch.no_grad():return model.field.density_fn(world).reshape(-1).float().cpu().numpy()
    sigma=np.concatenate([query(fit[i:i+65536]) for i in range(0,len(fit),65536)])
    if not np.isfinite(sigma).all() or np.median(sigma)<=0:raise ValueError('Nonpositive/nonfinite anchor isovalue')
    iso=float(np.median(sigma));bounds=np.stack((xyz.min(0),xyz.max(0)));margin=.1*(bounds[1]-bounds[0]).max()
    bounds+=np.array([[-margin],[margin]]);pitch=float((bounds[1]-bounds[0]).max()/(resolution-1))
    shape=np.ceil((bounds[1]-bounds[0])/pitch).astype(int)+1
    output.mkdir(parents=True,exist_ok=False);grid=np.empty(tuple(shape),np.float32);flat=grid.reshape(-1)
    for begin in range(0,len(flat),65536):
        stop=min(begin+65536,len(flat));ijk=np.stack(np.unravel_index(np.arange(begin,stop),tuple(shape)),-1)
        flat[begin:stop]=query(bounds[0]+ijk*pitch)
        if begin//65536%32==0:atomic_json(output/'progress.json',dict(status='density grid',completed=stop,total=len(flat),resolution=shape.tolist()))
    vertices,faces,_,_=marching_cubes(grid,level=iso,spacing=(pitch,)*3,allow_degenerate=False)
    vertices+=bounds[0];mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices.astype(np.float64)),o3d.utility.Vector3iVector(faces))
    o3d.io.write_triangle_mesh(str(output/'mesh.ply'),mesh)
    if fingerprint(model)!=before:raise ValueError('Extraction mutated field')
    atomic_json(output/'complete.json',dict(extraction='single scalar-field isosurface; no fusion of view-conditioned depths',
        checkpoint_sha256=sha(checkpoint),dataset_sha256=sha(data/'transforms.json'),anchors_receipt_sha256=sha(anchors.with_suffix('.json')),
        isovalue=iso,isovalue_rule='median native sigma at fitting anchors only',sigma_anchor_quantiles=np.quantile(sigma,[0,.1,.5,.9,1]).tolist(),
        grid_shape=shape.tolist(),grid_pitch_calibration_units=pitch,bounds_calibration=bounds.tolist(),
        vertices=len(vertices),triangles=len(faces),mesh_sha256=sha(output/'mesh.ply'),component_pruning=False,
        frozen_state_unchanged=True,uses_validation_rgb=False,uses_test_rgb=False,implementation_sha256=sha(__file__)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data','training','checkpoint','anchors','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--resolution',type=int,default=384);a=p.parse_args()
    extract(a.data,a.training,a.checkpoint,a.anchors,a.output,a.resolution)

if __name__=='__main__':main()
