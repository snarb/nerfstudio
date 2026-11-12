"""Render an exact-K flying-camera sample; frozen models and no training."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def supported_components(arrays, row, root, out):
    """Reject detached opaque islands unsupported by the GC surface cloud.

    This is a component-level 3D check, not a per-pixel GC silhouette: occluded
    backs and limbs remain eligible when their connected body has support.
    """
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree
    from flying_path import points
    from render_depth_cleanup import depth_support
    inventory = {c['name']: c for c in json.loads((out/'camera_inventory.json').read_text())['cameras']}
    cloud = points(root, inventory, row['source_frame'])
    tree = cKDTree(cloud)
    core, _, _ = depth_support(arrays, 10000, False, 120)
    labels, count = ndi.label(core)
    pose = np.array(row['pose']); fx, fy, cx, cy = row['intrinsics']
    rejected = []
    for label in range(1, count+1):
        y, x = np.nonzero(labels == label)
        y, x = y[::20], x[::20]
        rays = np.c_[(x+.5-cx)/fx, -(y+.5-cy)/fy, -np.ones(len(x))]
        rays /= np.linalg.norm(rays, axis=1)[:, None]
        world = pose[:3, 3] + (rays @ pose[:3, :3].T) * arrays['depth'][y, x]
        median_m = float(np.median(tree.query(world)[0]) / (.2217203609534452*1.0805))
        if median_m > .35:
            core[labels == label] = False
            rejected.append(dict(label=label, median_distance_m=median_m))
    if not core.any():
        raise ValueError('3D support rejected every component; review required')
    distance, nearest = ndi.distance_transform_edt(~core, return_indices=True)
    reference = arrays['depth'][..., 0][tuple(nearest)]
    return distance.astype(np.float32), reference.astype(np.float32), rejected


def render(args):
    root,out=Path(args.root),Path(args.output);sys.path.insert(0,str(root/'tools'))
    from inference import load_pipeline
    from common import sha,atomic
    from render_depth_cleanup import depth_support,install_depth_gate
    path=json.loads((out/'path.json').read_text());row=path['records'][args.index]
    assert row['kind']=='nerf'
    frame=row['source_frame'];storage=json.loads((root/'sparse24_storage_audit.json').read_text())['results'];saved=next(r for r in storage if r['frame']==frame)
    checkpoint=Path(saved['path']);assert sha(checkpoint)==saved['model_sha256']
    folder=out/'neural'/f'{args.index:03}';folder.mkdir(parents=True,exist_ok=True)
    started=time.time();pipe,spec,step=load_pipeline(root,checkpoint.parent.parent/'train/spec.json',checkpoint,folder)
    from nerfstudio.cameras.cameras import Cameras,CameraType
    K=row['intrinsics'];camera=Cameras(camera_to_worlds=torch.tensor(row['pose'],dtype=torch.float32)[None,:3,:],
        fx=K[0],fy=K[1],cx=K[2],cy=K[3],width=1920,height=1080,camera_type=CameraType.PERSPECTIVE).to(pipe.model.device)
    pipe.model.config.adaptive_max_frequency_level=10
    prior=json.loads((folder/'receipt.json').read_text()) if (folder/'receipt.json').exists() else {}
    if prior.get('path_sha256')==sha(out/'path.json') and prior.get('checkpoint_sha256')==saved['model_sha256'] and (folder/'raw.npz').exists():
        with np.load(folder/'raw.npz') as cached:
            arrays={k:cached[k].copy() for k in ('rgb','depth','accumulation')}
        raw={k:torch.from_numpy(v) for k,v in arrays.items()}
    else:
        with torch.inference_mode():raw=pipe.model.get_outputs_for_camera(camera)
        arrays={k:raw[k].float().cpu().numpy() for k in ('rgb','depth','accumulation')}
        np.savez_compressed(folder/'raw.npz',**arrays)
    distance,reference,rejected=supported_components(arrays,row,root,out)
    # New view => new support. Never reuse the prior GC silhouette.
    handle=install_depth_gate(pipe.model,camera,reference,distance,.15,.10,3,5)
    try:
        with torch.inference_mode():clean=pipe.model.get_outputs_for_camera(camera)
    finally:handle.remove()
    for name,result in [('raw',raw),('strict',clean)]:
        rgb=result['rgb'].float().cpu().numpy();assert np.isfinite(rgb).all()
        Image.fromarray(np.uint8(np.clip(rgb*255+.5,0,255))).save(folder/(name+'.png'))
    np.savez_compressed(folder/'strict.npz',**{k:clean[k].float().cpu().numpy() for k in ('rgb','depth','accumulation')})
    atomic(folder/'receipt.json',dict(index=args.index,frame=frame,checkpoint_sha256=saved['model_sha256'],step=step,
        camera=row,path_sha256=sha(out/'path.json'),renderer_sha256=sha(Path(__file__)),seconds=time.time()-started,source_pre_resize=False,strict_view_specific=True,rejected_components=rejected))
    print('DONE',args.index,frame,round(time.time()-started,2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--index',type=int,required=True);render(p.parse_args())
