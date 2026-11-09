#!/usr/bin/env python3
"""Add an oriented head-only density scope to an already cloned native parent."""
from pathlib import Path
import argparse,json,shutil
import numpy as np,torch,yaml
from probe_dec5_head_layer import head_bounds
from run_dec5_hair_campaign import atomic_json,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['parent','data','anchors','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    configs=list(a.parent.rglob('config.yml'));natives=list(a.parent.rglob('step-*.ckpt'))
    if len(configs)!=1 or len(natives)!=1:raise ValueError('Use one pinned cloned native checkpoint')
    cfg=yaml.load(configs[0].read_text(),Loader=yaml.Loader)
    if not cfg.pipeline.model.separate_density_encoding:raise ValueError('Clone density with matching Adam state first')
    cfg.pipeline.datamanager.data=a.data;cfg.pipeline.datamanager.dataparser.data=a.data
    payload=json.loads((a.data/'transforms.json').read_text());by={Path(f['file_path']).name:f for f in payload['frames']}
    parsed=cfg.pipeline.datamanager.dataparser.setup().get_dataparser_outputs(split='train')
    frames=[by[Path(p).name] for p in parsed.image_filenames];raw=np.asarray([f['transform_matrix'] for f in frames]);normal=parsed.cameras.camera_to_worlds.cpu().numpy()
    R=normal[0,:3,:3]@raw[0,:3,:3].T;s=np.linalg.norm(normal[1,:3,3]-normal[0,:3,3])/np.linalg.norm(raw[1,:3,3]-raw[0,:3,3]);t=normal[0,:3,3]-s*R@raw[0,:3,3]
    if np.linalg.norm(s*raw[:,:3,3]@R.T+t-normal[:,:3,3],axis=-1).max()>1e-4:raise ValueError('Camera normalization mismatch')
    bounds,coverage=head_bounds(a.anchors,frames);span=bounds[1]-bounds[0]
    matrix=np.diag(1/span)@R.T/s;bias=-matrix@t-bounds[0]/span;transform=np.c_[matrix,bias]
    cfg.pipeline.model.density_geometry_roi_transform=tuple(tuple(float(v) for v in row) for row in transform)
    (a.output/'config.yml').write_text(yaml.dump(cfg));native=torch.load(natives[0],map_location='cpu',weights_only=False)
    step=native['step'];name='gate_%06d.pt'%step
    for source,target in [(natives[0],a.output/'native'/natives[0].name),(a.parent/name,a.output/name)]:
        state=torch.load(source,map_location='cpu',weights_only=False);key='_model.field.density_geometry_roi_transform'
        if key in state['pipeline']:raise ValueError('Parent already has a density scope')
        state['pipeline'][key]=torch.tensor(transform,dtype=torch.float32);target.parent.mkdir(parents=True,exist_ok=True);torch.save(state,target)
    for name in ['request.json','scene_bounds.json','history.json']:shutil.copy2(a.parent/name,a.output/name)
    atomic_json(a.output/'head_scope.json',dict(bounds_calibration=bounds.tolist(),world_to_unit_head=transform.tolist(),
        affine_convention='unit = model_world @ matrix.T + bias',bounds_rule='fit hair anchors min/max, 10 percent largest-extent margin',
        withheld_anchor_coverage=coverage,anchors_sha256=sha(a.anchors),dataset_sha256=sha(a.data/'transforms.json'),
        native_parent_sha256=sha(natives[0]),original_weights_optimizer_scheduler_scaler_rng_preserved=True,implementation_sha256=sha(__file__)))

if __name__=='__main__':main()
