"""Prepare matched actor confidence configs; never construct a trainer or launch.

Final actor geometry and every production review remain explicit false gates.
The provisional source may be used for zero-update data/math probes only.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path('/fsx/tmp/marshawn_001660_hd')
sys.path[:0]=[str(ROOT/'code_native'),str(ROOT/'code_sfm/LookCloser/scripts'),
    str(ROOT/'final_pair/code'),str(Path(__file__).parent)]

def stage(manifest,output,bounds_source,bounds_name,geometry_manifest=None):
    import yaml
    from marshawn_actor_training_pipeline import (file_sha,ActorConfidencePipeline,
        ActorConfidenceDataManager,ActorConfidenceModel,ActorProposalPixelSampler)
    manifest=Path(manifest).resolve();payload=json.loads(manifest.read_text())
    if payload.get('source_RGB_unchanged')is not True or payload.get('no_alpha_GT')is not True:
        raise ValueError('Unchanged RGB/confidence-only source contract required')
    if payload.get('disqualified',False):raise ValueError('Disqualified provisional input')
    bounds_source=Path(bounds_source).resolve();bounds_payload=json.loads(bounds_source.read_text())
    if bounds_payload.get('kind')!='ROOT_REQUESTED_FIRST_EXPERIMENTAL_SAFETY_AABB_NOT_FINAL_MINIMUM':
        raise ValueError('Explicit ROOT-frozen PROGRAMMATIC actor bounds required; old subject/manual boxes frozen')
    bounds=bounds_payload['aabb']
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    code=output/'code';code.mkdir()
    helpers=['marshawn_actor_training_pipeline.py','marshawn_actor_confidence.py','marshawn_actor_metrics.py',
        'marshawn_actor_geometry.py','marshawn_true_los.py',
        'marshawn_packed_distortion_candidate.py','marshawn_crash_capture.py','marshawn_all_view_early_stop.py',
        'run_marshawn_actor_cold.py','run_marshawn_png_distortion_control.py']
    for name in helpers:shutil.copyfile(Path(__file__).parent/name,code/name)
    base_path=ROOT/'native_png_radialK4_HASH23_R0_COLD_STAGED/config.yml'
    base=yaml.load(base_path.read_text(),Loader=yaml.Loader)
    source_paths=[*code.glob('*.py'),ROOT/'code_native/nerfstudio/models/instant_ngp.py',
        ROOT/'code_native/nerfstudio/fields/nerfacto_field.py',ROOT/'code_native/nerfstudio/data/pixel_samplers.py',
        ROOT/'code_native/nerfstudio/pipelines/dynamic_batch.py',ROOT/'code_native/nerfstudio/engine/trainer.py',
        ROOT/'code_native/LookCloser/scripts/run_bounded_ngp_quiet.py',
        ROOT/'code_sfm/LookCloser/scripts/run_marshawn_hd_campaign.py',
        ROOT/'code_sfm/nerfstudio/models/lookcloser.py',bounds_source]
    for policy in ['hard','W32','W64']:
        folder=output/policy;folder.mkdir();cfg=copy.deepcopy(base);p=cfg.pipeline;dm=p.datamanager
        for name in list(vars(p)):
            if name.startswith(('final_','MVS_','packed_distortion_','floor_band_')):delattr(p,name)
        p._target=ActorConfidencePipeline;p.actor_manifest=manifest;p.actor_geometry_manifest=geometry_manifest
        p.actor_geometry_variant='actor_AUTO128_SAFETY';p.actor_preparation=ROOT/'data/preparation.json'
        p.actor_free_weight=.01;p.actor_band_weight=.01;p.actor_absence_manifest=None
        p.actor_distortion_source=ROOT/'code_sfm/nerfstudio/models/lookcloser.py'
        p.actor_distortion_source_sha256=file_sha(p.actor_distortion_source)
        p.actor_distortion_weight=.01;p.actor_eval_interval=2000;p.actor_grid_artifact_dir=folder/'grids'
        p.model._target=ActorConfidenceModel;p.model.background_color='black';p.model.render_step_size=.0005
        p.model.raw_no_appearance_embedding=True;p.model.use_appearance_embedding=False
        dm._target=ActorConfidenceDataManager;dm.actor_manifest=manifest;dm.actor_policy=policy
        dm.pixel_sampler._target=ActorProposalPixelSampler;dm.pixel_sampler.ignore_mask=False
        dm.dataparser.marshawn_aabb=bounds;dm.dataparser.data=Path(payload['data_dir']);dm.data=Path(payload['data_dir'])
        cfg.output_dir=folder/'run';cfg.experiment_name='marshawn_001660_actor_'+policy;cfg.timestamp='cold_seed42'
        config=folder/'config.yml';config.write_text(yaml.dump(cfg))
        # Re-importable public targets, no <locals> serialization.
        restored=yaml.load(config.read_text(),Loader=yaml.Loader)
        assert restored.pipeline._target is ActorConfidencePipeline
        spec=dict(kind='ACTOR_CONFIDENCE_COLD_PREPARATION_NOT_AUTHORIZED',config_sha256=file_sha(config),
            actor_manifest=str(manifest),actor_manifest_sha256=file_sha(manifest),
            sources_sha256={str(path):file_sha(path)for path in source_paths},
            aabb_prepared_world=bounds,aabb_proposal_source=str(bounds_source),aabb_proposal_name=bounds_name,
            aabb_proposal_sha256=file_sha(bounds_source),aabb_experimental_safety_control_not_final_minimum=True,policy=policy,max_attempts=22000,
            attempt_indices='0..21999 inclusive; first2000 checkpoint has2001 attempts',
            first_required_pause_before=2001,selection_receipt=str(folder/'selection.json'),
            shared_proposal='GLOBALuniform W64 pixels across TRAIN63; fixed global policy mean, weights once',
            RGB_target='original corrected RGB3; never premultiplied or alpha-labelled',
            geometry_policy='PENDING vetted actorSfM and >256-native semantic-absence review; no fullscene assets',
            proposed_common_config=dict(background='black',hash_log2=23,step_world=.0005,grid_radius=0),
            release_checks={k:False for k in ['all66_final_data','actor_aabb','actor_geometry','actual_zeroopt','peer','root_training_authorized']})
        sp=folder/'run_spec.json';sp.write_text(json.dumps(spec,indent=2)+'\n')
        argv=['/home/pavlo/.conda/envs/nerfstudio/bin/python',str(code/'run_marshawn_actor_cold.py'),'controller',
            '--config',str(config),'--spec',str(sp),'--data',str(dm.dataparser.data),'--output-dir',str(cfg.output_dir),
            '--experiment-name',cfg.experiment_name,'--timestamp',cfg.timestamp,'--scene-scale','4',
            '--center-method','none','--orientation-method','none','--scale-factor','1',
            '--max-num-iterations','22000','--step-interval','2000','--render-step-size','.0005',
            '--eval-num-rays-per-chunk','512','--no-update-summary','--eval-checkpoint','best','--poll-seconds','30']
        env=dict(PYTHONPATH=':'.join(map(str,[code,ROOT/'code_native',ROOT/'code_native/LookCloser/scripts',
            ROOT/'code_sfm/LookCloser/scripts',ROOT/'final_pair/code'])),
            TORCH_EXTENSIONS_DIR=str(ROOT/'native_ngp4_corrected_sdr/local_torch_extensions_py311_cu128_sm120_ptx'),
            TORCH_CUDA_ARCH_LIST='12.0+PTX',MAX_JOBS='2',PYTHONDONTWRITEBYTECODE='1',OPENCV_IO_ENABLE_OPENEXR='1')
        (folder/'launch_plan.json').write_text(json.dumps(dict(command=argv,env=env,status='NOT_AUTHORIZED_PREPARATION_ONLY',
            checkpoint_scope='cold actor-only; no fullscene checkpoint load',no_training_launched=True),indent=2)+'\n')
    (output/'preparation.json').write_text(json.dumps(dict(manifest=str(manifest),manifest_sha256=file_sha(manifest),
        sources_copied=True,optimizer_updates=0,training_launched=False,
        caveat='Same confidence ablation only after ONE geometry/BG/dt/hash config and final66/native semantic domains approved'),indent=2)+'\n')
    print(output)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--bounds-source',type=Path,required=True)
    parser.add_argument('--bounds-name',default='actor_AUTO128_SAFETY');parser.add_argument('--geometry-manifest',type=Path)
    args=parser.parse_args();stage(args.manifest,args.output,args.bounds_source,args.bounds_name,args.geometry_manifest)
