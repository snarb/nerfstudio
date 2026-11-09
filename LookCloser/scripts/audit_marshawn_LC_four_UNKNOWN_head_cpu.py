"""Verify actual fit-only LA UNKNOWN counterfactual and compare immutable control."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from marshawn_frozen_semantic_head import FrozenActorGrid,tensor_digest
from run_marshawn_frozen_semantic_probe import nested_digest
from prepare_marshawn_frozen_semantic_cpu import sha,write


def run(c,archive):
    torch.set_num_threads(2)
    control=c/'frozen_LC_semantic_head_archives/recovery1_20261107_v1/runs/prior001_fresh2064/head.pt'
    checkpoint=archive/'runs/prior001_fresh2064/head.pt';complete_path=archive/'runs/prior001_fresh2064/complete.json'
    complete=json.loads(complete_path.read_text());initial=json.loads((archive/'runs/prior001_fresh2064/initial_state.json').read_text())
    labels=archive/'label_qualification.json';spec=archive/'spec.json';release=archive/'root_release.json'
    if sha(labels)!='44e7734ed93a2c10742de6dc55886978d7c2097ee62779915dada93feda94df4'or sha(spec)!='ccfdd894d29d06ef17b8dffd8fa66b7db06b4a8383d7691246632d8fff7fba93':raise ValueError('Exact four-UNKNOWN metadata required')
    if sha(release)!='4895b50091815d6f5abadd1e9e7a07fa2ec811716c95dcde9000985088b3073e'or sha(checkpoint)!=complete['checkpoint_sha256']:
        raise ValueError('Actual root release/head bytes differ')
    old,new=[torch.load(p,map_location='cpu',weights_only=False)for p in [control,checkpoint]]
    if (new['iteration']!=2064 or complete['iteration']!=2064 or complete['actual_head_updates']!=2064 or initial['initial_iteration']!=0 or initial['initial_Adam_count']!=0 or initial['restoration']is not None or initial['qualified_unknown_queries']!=4 or
            new['label_qualification_sha256']!=sha(labels)or new['spec_sha256']!=sha(spec)or new['initial_probability']!=.01 or
            new['source_kind']!='DIRECT_FROZEN_LC_NATIVE_VISIBILITY'or new['model_digest']!=old['model_digest']or new['LC_capture_receipt_sha256']!=old['LC_capture_receipt_sha256']or not new['world_to_box_is_identity']or
            new['camera_history']!=old['camera_history']or not complete['native_checkpoint_and_caches_unchanged']or not complete['native_model_not_loaded_or_optimized']):
        raise ValueError('Controlled fresh2064 source/history/UNKNOWN identity differs')
    if not all(int(v['step'])==2064 and torch.isfinite(v['exp_avg']).all()and torch.isfinite(v['exp_avg_sq']).all()for v in new['optimizer']['state'].values()):
        raise ValueError('Actual Adam state/count invalid')
    def head(state):
        result=FrozenActorGrid(state['head']['bounds'],np.eye(4),128,initial_probability=.01);result.load_state_dict(state['head'],strict=True);result.eval();result.requires_grad_(False)
        if tensor_digest(result)!=state['head_digest']or not torch.equal(result.world_to_box,torch.eye(4)):raise ValueError('Head exact state/frame differs')
        return result
    prior,current=head(old),head(new)
    trace=c/'frozen_LC_semantic_physical_plan_v3/prior001/renders/DC_trace_1097_572.npz'
    with np.load(trace)as z:point=z['xyz'][int(np.argmax(z['hard_theta05_selected_weight']))].copy()
    witnesses_path=c/'semantic_cell_candidates_cpu_v1/prior001_fresh2064/hard_theta05/witnesses.json'
    witnesses=json.loads(witnesses_path.read_text());points=np.asarray([r['witness_xyz']for r in witnesses['records']],np.float32)
    with torch.no_grad():
        actual_point=[float(h(torch.from_numpy(point[None]))[0])for h in [prior,current]]
        point_values=[h(torch.from_numpy(points)).numpy()for h in [prior,current]]
    changed=[]
    for i,(before,after)in enumerate(zip(*point_values)):
        if (before>=.1)!=(after>=.1)or(before>=.5)!=(after>=.5):changed.append(dict(**witnesses['records'][i],old_probability=float(before),new_probability=float(after)))
    delta=(new['head']['logits']-old['head']['logits']).double().numpy()
    report=dict(CPU_only=True,CUDA_initialized=torch.cuda.is_initialized(),optimizer_created=False,fresh_actual_Adam_steps=2064,
        qualified_UNKNOWN_count=4,original_cache_UNKNOWN_count=2,exact_camera_history_control_match=True,
        final_RNG_and_sampler_exact=nested_digest(new['rng'])==nested_digest(old['rng']),new_head_sha256=sha(checkpoint),new_head_digest=new['head_digest'],
        control_head_sha256=sha(control),source_pipeline_digest=new['model_digest'],source_capture_receipt_sha256=new['LC_capture_receipt_sha256'],
        actual_dominant_DC_point=point.tolist(),actual_point_old_new_class_probability=actual_point,
        original109_reference_observations=dict(old_min=float(point_values[0].min()),new_min=float(point_values[1].min()),
            retained_old_new_ge01=[int((p>=.1).sum())for p in point_values],retained_old_new_ge05=[int((p>=.5).sum())for p in point_values],threshold_crossings=changed),
        logit_difference=dict(max_abs=float(np.abs(delta).max()),RMSE=float(np.sqrt(np.mean(delta**2))),changed_nodes=int((np.abs(delta)>1e-5).sum())),
        source_and_native_cache_unchanged=True,canonical_targets_andRGBunchanged=True,
        bindings={str(p):sha(p)for p in [checkpoint,control,complete_path,labels,spec,release,trace,witnesses_path]},
        caveat='Head-point probabilities and expected semantic support only; no physicalopacity/emptyspace/footpreservation certification without separate actual3FHD/nativeσ-selected render. No threshold changed or automatic promotion.')
    write(archive/'CPU_four_UNKNOWN_control_audit.json',report);print(json.dumps({k:v for k,v in report.items()if k not in ['bindings','original109_reference_observations']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--archive',type=Path,required=True)
    args=parser.parse_args();run(args.campaign,args.archive)
