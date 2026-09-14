"""Matched real-train neck crops for the observed-depth notch pilot."""
from pathlib import Path
from copy import deepcopy
import argparse
import numpy as np
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,exr,display,ROOT
from study_jaw_boundary_notches import PARENT


def run(root,frame):
    import render_smooth_temporal_mesh_video as renderer
    from temporal_texture_view_prior import install
    from wide_dynamic_camera_flight import install_source_masks
    renderer.torch.set_num_threads(2);install(renderer);install_source_masks(renderer)
    source=Path('/mnt/data/dec5_jaw_3d_train_evidence/request.json');crop_spec=read(source)
    profiles=read(ROOT/'camera_profiles.json');gains=dict(zip(profiles['physical_cameras'],profiles['rgb_gain']))
    exposure=read(ROOT/'exposure.json')['fixed_exposure_gain'];records=[]
    for name in ['F004_E005_1210FP','M004_B005_12109O']:
        spec=next(r for r in crop_spec['records'] if r['frame']==frame and r['camera']['physical_camera']==name)
        camera=spec['camera'];panels=[]
        if sha(camera['file_path'])!=spec['source_sha256']:raise ValueError('Real train RGB changed')
        gt=np.rint(display(exr(camera['file_path'])*np.array(gains[name]),exposure)*255).clip(0,255).astype(np.uint8)
        panels.append(Image.fromarray(np.rot90(gt)).crop(spec['crop']))
        for variant in ['baseline','guarded']:
            out=root/'train_rgb'/frame/name/variant;request=deepcopy(renderer.verify_request(PARENT))
            request['inventory']=[r for r in request['inventory'] if r['frame_id']==frame]
            row=request['inventory'][0];row['camera']=camera
            if variant=='guarded':
                result=read(root/'guarded'/frame/'result.json')
                if not result['observed_free_space_guard_passed']:raise ValueError('Failed geometry guard')
                row['mesh']=str(root/'guarded'/frame/'mesh.ply');row['mesh_sha256']=sha(row['mesh'])
                if row['mesh_sha256']!=result['mesh_sha256']:raise ValueError('Changed candidate')
                row['jaw_guard_result_sha256']=sha(root/'guarded'/frame/'result.json')
            request.update(partial_diagnostic_only=True,full_video_candidate=False,real_train_validation=True,
                           train_crop_source_sha256=sha(source),target_camera=name)
            request['script_hashes'][Path(__file__).name]=sha(__file__)
            out.mkdir(parents=True,exist_ok=True);(out/'frames').mkdir(exist_ok=True)
            if (out/'request.json').exists() and read(out/'request.json')!=request:raise ValueError('Frozen validation mismatch')
            atomic_json(out/'request.json',request);renderer.render(out,[frame])
            path=out/'frames'/frame/'frame.png';panels.append(Image.open(path).convert('RGB').crop(spec['crop']))
        panel=Image.new('RGB',(960,350));draw=ImageDraw.Draw(panel)
        for i,image in enumerate(panels):panel.paste(image,(i*320,30));draw.text((i*320+4,5),['real train RGB','baseline','guarded'][i],fill='white')
        folder=root/'train_review'/frame;folder.mkdir(parents=True,exist_ok=True);path=folder/f'{name}.png';panel.save(path)
        records.append(dict(camera=name,crop=spec['crop'],path=str(path),sha256=sha(path),gt_source_sha256=spec['source_sha256']))
    atomic_json(root/'train_review'/frame/'receipt.json',dict(records=records,visual_status='pending',
        profile_sha256=sha(ROOT/'camera_profiles.json'),exposure_sha256=sha(ROOT/'exposure.json'),
        training_views_not_heldout_metrics=True,used_for_fitting=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/mnt/data/dec5_jaw_measured_depth'))
    p.add_argument('--frame',required=True,choices=['001193','001195']);args=p.parse_args();run(args.root,args.frame)
