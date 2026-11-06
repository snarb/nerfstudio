"""Bounded real-source CPU checks for staged native resolution validation."""
import argparse,copy,hashlib,importlib.util,json,os,sys,tempfile
from pathlib import Path
from types import SimpleNamespace


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def main(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    root=a.plan.parent.parent.parent;native=root/'native_png_radialK4_COLD_R1_DEV3_STAGED';code=a.plan/'adapter/code'
    sys.path[:0]=[str(code),str(native/'code'),str(root/'code_native')]
    import numpy as np,torch,yaml
    from marshawn_native_resolution_binding import validate_native_resolution_binding
    torch.set_num_threads(2);config=yaml.load((native/'config.yml').read_text(),Loader=yaml.BaseLoader)['pipeline']
    old=root/'color_variants/radial_k4_joint_scalar_DIAGNOSTIC/sdr_png/transforms.json';new=a.plan/'proposed_transforms_NATIVE_WORLD_NOT_RELEASED.json';oldsha=sha(old)
    identity=validate_native_resolution_binding(old,old,oldsha,1);high=validate_native_resolution_binding(new,old,oldsha,2)
    negative=[];document=json.loads(new.read_text())
    with tempfile.TemporaryDirectory(prefix='native_resolution_negative_') as td:
        for name in ['floor_pose','wrong_focal','eval_scale','split_order','geometry_hash']:
            changed=copy.deepcopy(document);expected=oldsha
            if name=='floor_pose':changed['frames'][0]['transform_matrix'][0][3]+=.001
            if name=='wrong_focal':changed['frames'][0]['fl_x']+=.01
            if name=='eval_scale':next(r for r in changed['frames'] if r['file_path'] in changed['val_filenames'])['w']*=2
            if name=='split_order':changed['frames'][3],changed['frames'][4]=changed['frames'][4],changed['frames'][3]
            if name=='geometry_hash':expected='0'*64
            path=Path(td)/(name+'.json');path.write_text(json.dumps(changed))
            try:validate_native_resolution_binding(path,old,expected,2)
            except ValueError:negative.append(name)
            else:raise AssertionError('Negative accepted '+name)
    # Real geometry arrays and private sampling are compared without a model.
    original_free=module('original_free_cpu',native/'code/marshawn_los_objective_frozen.py')
    staged_free=module('staged_free_cpu',code/'marshawn_los_objective_frozen.py')
    original_band=module('original_band_cpu',native/'code/marshawn_surface_band_objective.py')
    staged_band=module('staged_band_cpu',code/'marshawn_surface_band_objective.py')
    bounds=np.asarray(config['datamanager']['dataparser']['marshawn_aabb'],float);args=[config['final_geometry_manifest'],config['final_geometry_variant'],bounds,old,config['final_geometry_preparation']]
    free=[m.FreeSpaceObjective(*args,weight=.01,rays=128,samples=64,seed=42,device='cpu') for m in [original_free,staged_free]]
    bands=[m.SurfaceBandObjective(*args,weight=.01,rays=128,samples=16,seed=42,device='cpu') for m in [original_band,staged_band]]
    for left,right in [free,bands]:
        aa,bb=dict(left.named_buffers()),dict(right.named_buffers());assert aa.keys()==bb.keys() and all(torch.equal(aa[k],bb[k]) for k in aa)
    names=identity['physical_camera_order'];filenames=[Path('train_'+name+'.png') for name in names]
    def pipeline(data):return SimpleNamespace(device='cpu',config=SimpleNamespace(datamanager=SimpleNamespace(dataparser=SimpleNamespace(data=data)),final_geometry_transforms=old),datamanager=SimpleNamespace(train_dataset=SimpleNamespace(image_filenames=filenames)))
    original_mvs=module('original_mvs_cpu',native/'code/marshawn_mvs_distribution_candidate.py');staged_mvs=module('staged_mvs_cpu',code/'marshawn_mvs_distribution_candidate.py');manifest=Path(*config['MVS_depth_manifest'])
    mvs=[original_mvs.MVSDistributionDepth(manifest,config['MVS_depth_manifest_sha256'],pipeline(old.parent)),staged_mvs.MVSDistributionDepth(manifest,config['MVS_depth_manifest_sha256'],pipeline(a.plan/'adapter/data_4k'))]
    for k in ['origins','directions','depth','normal','camera_indices']:assert torch.equal(getattr(mvs[0],k),getattr(mvs[1],k)),k
    steps=[20000,30000,35000,100000];samples=[]
    for step in steps:
        aa,bb=[f.sample_batch(step) for f in free];assert all(torch.equal(aa[k],bb[k]) for k in aa)
        ab,bb=[f.sample_batch(step,aa['indices']) for f in bands];assert all(torch.equal(ab[k],bb[k]) for k in ab)
        ma,mb=[f.sample(step) for f in mvs];assert all(torch.equal(x,y) if torch.is_tensor(x) else x==y for x,y in zip(ma,mb))
        samples.append(dict(step=step,SfM_ids_and_points_exact=True,surface_ids_and_points_exact=True,MVS_ids_and_private_seed_exact=True))
    assert not torch.cuda.is_initialized()
    result=dict(identity_binding=identity,high_binding=high,rejected_negative_cases=negative,all_geometry_buffer_tensors_exact=True,private_sampling_checks=samples,no_GPU_model_or_optimizer=True,constructor_source_changes_only=True,helper_sha256=sha(__file__),stage_receipt_sha256=sha(a.plan/'adapter/stage_receipt.json'),limitations=['Actual native prior outputs/gradients and checkpoint/runtime replay remain technical GPU gates after review.'])
    (a.plan/'adapter/CPU_equivalence.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(negative_cases=negative,geometry_sampling_steps=steps,no_GPU=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);main(p.parse_args())
