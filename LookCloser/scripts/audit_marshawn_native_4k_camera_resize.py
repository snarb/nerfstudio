"""CPU-only actual Cameras resize/ray parity and unchanged native compositing-loop proof."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
import numpy as np


def main(a):
    for path in reversed(a.python_path):sys.path.insert(0,str(path))
    import torch
    from nerfstudio.cameras.cameras import Cameras,CameraType
    diagnostic=json.loads(a.diagnostic.read_text());native=json.loads(a.native.read_text());lookup={r['physical_camera']:r for r in native['frames']};rows=[]
    assert diagnostic['training_eligible']is False
    for row in diagnostic['frames']:
        original=lookup[row['physical_camera']]
        camera=Cameras(camera_to_worlds=torch.tensor([original['transform_matrix']],dtype=torch.float32)[:,:3],fx=original['fl_x'],fy=original['fl_y'],cx=original['cx'],cy=original['cy'],width=original['w'],height=original['h'],camera_type=CameraType.PERSPECTIVE)
        resized=copy.deepcopy(camera);resized.rescale_output_resolution(2.)
        coords=torch.tensor([[.5,.5],[row['h']/2+.5,row['w']/2+.5],[row['h']-.5,row['w']-.5]],dtype=torch.float32)
        old=camera.generate_rays(0,coords=coords/2,keep_shape=False);new=resized.generate_rays(0,coords=coords,keep_shape=False)
        errors=dict(origin=float(torch.max(torch.abs(old.origins-new.origins))),direction=float(torch.max(torch.abs(old.directions-new.directions))))
        assert max(errors.values())<1e-6
        ratios={key:float(getattr(resized,key).item()/getattr(camera,key).item())for key in['fx','fy','cx','cy','width','height']};assert all(v==2 for v in ratios.values())
        inverse=np.linalg.inv(np.asarray(diagnostic['world_to_box']))@row['transform_matrix'];pose_error=float(np.max(np.abs(inverse-original['transform_matrix'])));assert pose_error<1e-12
        rows.append(dict(camera=row['physical_camera'],actual_CPU_camera_resize_ratios=ratios,continuous_ray_errors=errors,floor_to_native_pose_error=pose_error))
    source=a.source.read_text();staged=a.staged.read_text();begin='    with torch.inference_mode():';end='    for name,buf in buffers.items():'
    assert source[source.index(begin):source.index(end)]==staged[staged.index(begin):staged.index(end)]
    result=dict(passed=True,CPU_only=True,no_model_loaded=True,rows=rows,native_sampling_field_AABB_partition_compositing_loop_byte_unchanged=True,
                bindings={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in[a.diagnostic,a.native,a.source,a.staged,Path(__file__)]},
                limitation='Continuous ray equivalence, not equality of FHD and4K discrete pixel samples or an image-parity claim.')
    with a.output.open('x')as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in['diagnostic','native','source','staged','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--python-path',type=Path,action='append',default=[]);main(p.parse_args())
