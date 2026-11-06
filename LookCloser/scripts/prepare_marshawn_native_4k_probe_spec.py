"""Prepare unreleased native resolution probe configs without importing CUDA."""
import argparse,copy,hashlib,json,shutil
from pathlib import Path
import yaml
from yaml.nodes import MappingNode,ScalarNode,SequenceNode


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def get(node,key):return next(value for name,value in node.value if name.value==key)


def put(node,key,value):
    for i,(name,prior) in enumerate(node.value):
        if name.value==key:node.value[i]=(name,value);return
    node.value.append((ScalarNode('tag:yaml.org,2002:str',key),value))


def scalar(value):
    if value is None:return ScalarNode('tag:yaml.org,2002:null','null')
    if isinstance(value,int):return ScalarNode('tag:yaml.org,2002:int',str(value))
    return ScalarNode('tag:yaml.org,2002:str',str(value))


def pathnode(path):return SequenceNode('tag:yaml.org,2002:python/object/apply:pathlib.PosixPath',[scalar(str(path))])


def differences(a,b,prefix=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return sum((differences(a.get(k),b.get(k),prefix+'.'+k) for k in sorted(set(a)|set(b))),[])
    return [] if a==b else [dict(key=prefix,before=a,after=b)]


def main(a):
    plan=a.plan;root=plan.parent.parent.parent;campaign=root/'tuning_20261106';adapter=plan/'adapter';code=adapter/'code';source=root/'color_variants/radial_k4_joint_scalar_DIAGNOSTIC/sdr_png';template=plan.parent/'native_continuation_plan/proposed_config.yml'
    node=yaml.compose(template.read_text());base=yaml.load(template.read_text(),Loader=yaml.BaseLoader);files={p.name:sha(p) for p in code.glob('*.py')};configs=[]
    for name,scale,document in [('HD_identity',1,json.loads((source/'transforms.json').read_text())),('TRAIN4K',2,json.loads((plan/'proposed_transforms_NATIVE_WORLD_NOT_RELEASED.json').read_text()))]:
        data=adapter/('data_'+name);data.mkdir(exist_ok=False)
        if scale==1:
            remap={}
            for row in document['frames']:
                old=row['file_path'];row['file_path']=str(source/old);row['mask_path']=str(source/row['mask_path']);remap[old]=row['file_path']
            for split in ['train_filenames','val_filenames','test_filenames']:document[split]=[remap[p] for p in document[split]]
        (data/'transforms.json').write_text(json.dumps(document,indent=2)+'\n');shutil.copy2(source/'color_contract.json',data/'color_contract.json')
        bound={str(Path(row[key])):sha(row[key]) for row in document['frames'] for key in ['file_path','mask_path']};bound['color_contract.json']=sha(data/'color_contract.json')
        gate=adapter/(name+'_RGB_GATE_NOT_APPROVED.json');gate.write_text(json.dumps(dict(approved=False,data=str(data),transforms_sha256=sha(data/'transforms.json'),files_sha256=bound,photometric_contract_reused_unchanged=True,source_plan_sha256=sha(plan/'source_parity.json'),technical_GPU_gate_requires_separate_root_review=True),indent=2)+'\n')
        cfg=copy.deepcopy(node);pipeline=get(cfg,'pipeline');dm=get(pipeline,'datamanager');parser=get(dm,'dataparser');put(dm,'data',pathnode(data));put(parser,'data',pathnode(data))
        put(cfg,'load_checkpoint',scalar(None));put(cfg,'output_dir',pathnode(adapter/'NOT_RELEASED_RUNTIME'/name));put(cfg,'experiment_name',scalar('native_resolution_'+name));put(cfg,'timestamp',scalar('anchor_PENDING_REVIEW'))
        for key,value in [('final_geometry_transforms',str(source/'transforms.json')),('final_geometry_transforms_sha256',sha(source/'transforms.json')),('final_rgb_train_scale',scale),('final_color_gate',str(gate)),('final_color_gate_sha256',sha(gate))]:put(pipeline,key,scalar(value))
        put(pipeline,'final_grid_artifact_dir',pathnode(adapter/'NOT_RELEASED_RUNTIME'/name/'grids'))
        put(pipeline,'final_source_hashes',MappingNode('tag:yaml.org,2002:map',[(scalar(k),scalar(v)) for k,v in sorted(files.items())]))
        config=adapter/(name+'_config_NOT_RELEASED.yml');config.write_text(yaml.serialize(cfg));parsed=yaml.load(config.read_text(),Loader=yaml.BaseLoader);diff=differences(base,parsed)
        # No scientific computation/sampling/optimizer fields change in templates.
        assert parsed['optimizers']==base['optimizers'] and parsed['pipeline']['model']==base['pipeline']['model']
        configs.append(dict(name=name,train_scale=scale,config=str(config),config_sha256=sha(config),RGB_gate=str(gate),RGB_gate_sha256=sha(gate),changes=diff))
    production=campaign/'native_reference_continuation_20000';state=json.loads((production/'state.json').read_text());selected=state['selection'];history=[dict(step=r['step'],checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256']) for r in state['evaluations']]
    spec=dict(kind='NATIVE_MIXED_RESOLUTION_TECHNICAL_PROBE_PROPOSAL',technical_GPU_gate_authorized=False,production_launch_authorized=False,source_checkpoint_NOT_FINAL={k:selected[k] for k in ['step','checkpoint','checkpoint_sha256']},must_refresh_source_after_HD_completion=True,source_requires=['completed commonV1 gate','checkpoint model/optimizer/scheduler/scaler/RNG','runtime dynamic-rays/cache order','authoritative native core/active sidecar','source FHD renders/metrics'],configs=configs,python_path=[str(code),str(root/'native_png_radialK4_COLD_R1_DEV3_STAGED/code'),str(root/'code_native'),str(root/'code_sfm/LookCloser/scripts'),str(plan.parent/'code')],GPU_exclusivity='Acquire same global GPU lock only after HD supervisor/worker exit and complete.json; never interrupt HD.',phases=[dict(name='HD_identity_restore',updates=0,checks=['all checkpoint domains bitwise','cache/dynamic/core/RNG exact','original vs adapted fixed-step prior tensors, values and gradients','commonFHD numeric parity']),dict(name='TRAIN4K_restore',updates=0,checks=['same domains and priors','all63 native world cameras/intrinsics','finite validRGB samples; original private prior sampling','commonFHD numeric parity','RSS/VRAM peak including cache collation']),dict(name='TRAIN4K_updates_and_restart',updates=32,save_after=16,checks=['actual Adam increment separately from attempts/AMP skips','save/reload all state after16','complete32 from exact saved checkpoint','compare next private sampling and RNG afterroundtrip','technical checkpoint excluded from production ancestry'])],render_parity_limits=dict(PNG_max_code=1,LPIPS_absolute=1e-5,PSNR_absolute_dB=.001),resource_policy=dict(expected_floatRGB_GiB=5.472220420837402,estimated_RGB_valid_two_copy_GiB=11.856483459472656,reserve_host_available_GiB=20,full_probe_after_HD_exit=True),control_history_preserved=history,matched_control='Reuse HD suffix only from exactly same selected source/runtime ancestry; otherwise separate reviewed controlled HD replay. Never prune suffix checkpoints.',selection='commonV1 minimumactorLPIPS; .001tie thenPSNR/earlierstep; unchanged source remains eligible',production_controller_pending='Rebase anchor to actual sourceS;5k gates and patience4, no22k cap/LR reset. Separate review required.',heldout4K='No4KEVAL source generated. Any later4K heldout visual diagnostic must remain distinct from primary FHD scoring.')
    (adapter/'technical_probe_spec.json').write_text(json.dumps(spec,indent=2)+'\n');print(json.dumps(dict(configs=len(configs),source_snapshot=selected['step'],GPU_authorized=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);main(p.parse_args())
