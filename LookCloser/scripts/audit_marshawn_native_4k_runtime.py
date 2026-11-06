"""Reproduce mixed-resolution camera parity using only the frozen CPU parser."""
import argparse,copy,hashlib,json,os,sys
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Run with CUDA_VISIBLE_DEVICES empty; do not import the full TrainerConfig')
    r=a.plan.parent.parent.parent;n=r/'native_png_radialK4_COLD_R1_DEV3_STAGED'
    sys.path[:0]=[str(r/'code_native'),str(r/'code_sfm/LookCloser/scripts')]
    import torch,yaml
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
    from run_marshawn_hd_campaign import MarshawnDataParser
    raw=yaml.load((n/'config.yml').read_text(),Loader=yaml.BaseLoader)['pipeline']['datamanager']['dataparser']
    old=NerfstudioDataParserConfig(_target=MarshawnDataParser,data=r/'color_variants/radial_k4_joint_scalar_DIAGNOSTIC/sdr_png',downscale_factor=1,eval_mode='filename',scene_scale=4.,orientation_method='none',center_method='none',auto_scale_poses=False,scale_factor=1.,load_3D_points=False)
    old.marshawn_aabb=[[float(v) for v in row] for row in raw['marshawn_aabb']];new=copy.deepcopy(old);new.data=a.plan/'proposed_transforms_NATIVE_WORLD_NOT_RELEASED.json';torch.set_num_threads(2);rows=[]
    for split,scale in [('train',2),('val',1)]:
        first=old.setup().get_dataparser_outputs(split);second=new.setup().get_dataparser_outputs(split)
        assert torch.equal(first.cameras.camera_to_worlds,second.cameras.camera_to_worlds)
        for key in ['fx','fy','cx','cy','width','height']:assert torch.equal(getattr(first.cameras,key)*scale,getattr(second.cameras,key)),key
        assert torch.equal(first.scene_box.aabb,second.scene_box.aabb)
        assert torch.equal(first.dataparser_transform,second.dataparser_transform) and first.dataparser_scale==second.dataparser_scale
        checks=[]
        for i in range(len(first.cameras)):
            coords=torch.tensor([[.5,.5],[505.5,959.5],[1011.5,1919.5]])
            aa=first.cameras.generate_rays(i,coords=coords);bb=second.cameras.generate_rays(i,coords=coords*scale)
            assert torch.equal(aa.origins,bb.origins) and torch.equal(aa.directions,bb.directions)
            checks.append(dict(index=i,name=first.image_filenames[i].stem,origins_max=float((aa.origins-bb.origins).abs().max()),directions_max=float((aa.directions-bb.directions).abs().max()),pixel_area_ratio=(bb.pixel_area/aa.pixel_area).flatten().tolist()))
        rows.append(dict(split=split,count=len(first.cameras),camera_tensors_exact_scaled=True,field_scene_box_exact=True,normalization_exact=True,rays=checks))
    assert not torch.cuda.is_initialized()
    result=dict(actual_frozen_parser_CPU=True,GPU_initialized=False,image_cache_created=False,rows=rows,helper_sha256=sha(__file__),native_config_sha256=sha(n/'config.yml'),proposed_transforms_sha256=sha(new.data),code_paths=sys.path[:2])
    a.output.write_text(json.dumps(result,indent=2)+'\n');print([(row['split'],row['count']) for row in rows])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
