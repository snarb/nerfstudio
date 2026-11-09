"""CPU diagnosis of actual unchanged-P2 GPU witness; no model or tolerance edits."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
import torch
from marshawn_stable_packed_density_reference import stable_packed_density_weights
from prepare_marshawn_frozen_semantic_cpu import sha,write


def run(witness,output,observer):
    torch.set_num_threads(2)
    completion_path=witness/'complete.json';completion=json.loads(completion_path.read_text())
    if not completion['reproduced']or not all(completion['source_unchanged'].values())or completion['reference_or_tolerance_changed']:
        raise ValueError('Actual unchanged-observer failed CUDA witness with immutable source required')
    archive=Path(completion['archive'])
    if sha(archive)!=completion['archive_sha256']or sha(observer)!=completion['unchanged_P2_observer_sha256']:
        raise ValueError('Witness or original observer changed')
    with np.load(archive)as z:arrays={k:z[k].copy()for k in z.files}
    output.mkdir(parents=True,exist_ok=False)
    starts,ends,sigma,ids=[arrays[k]for k in ['starts','ends','sigma','ids']]
    tau=sigma.astype(np.float64)*(ends.astype(np.float64)-starts.astype(np.float64))
    ray_count=len(arrays['native_rgb']);trans=np.ones(ray_count,np.float64);serial=np.empty(len(ids),np.float64);local_prefix=np.empty(len(ids),np.float64)
    optical_before=np.zeros(ray_count,np.float64)
    for i,(depth,ray)in enumerate(zip(tau,ids)):
        local_prefix[i]=optical_before[ray]
        serial[i]=trans[ray]*(-math.expm1(-depth));trans[ray]*=math.exp(-depth);optical_before[ray]+=depth
    stable=stable_packed_density_weights(*[torch.from_numpy(v)for v in [starts,ends,sigma,ids]],ray_count).numpy()
    loader=importlib.util.spec_from_file_location('_CPU_original_P2_witness_reference',observer);old=importlib.util.module_from_spec(loader);loader.loader.exec_module(old)
    old_CPU=old.packed_density_weights(*[torch.from_numpy(v)for v in [starts,ends,sigma,ids]],ray_count).numpy()
    global_cumsum=np.cumsum(tau);global_subtract_current=global_cumsum-tau
    first=np.r_[True,ids[1:]!=ids[:-1]];bases=np.zeros(ray_count);bases[ids[first]]=global_subtract_current[first]
    old_exclusive=global_subtract_current-bases[ids]
    # Isolate the two unstable mechanisms without accepting either as a reference.
    global_shift=np.r_[0.,global_cumsum[:-1]];shift_base=np.zeros(ray_count);shift_base[ids[first]]=global_shift[first]
    global_shift_subtract_base=global_shift-shift_base[ids]
    local_subtract_current=np.empty_like(tau)
    for ray in np.unique(ids):
        choose=ids==ray;local_subtract_current[choose]=np.cumsum(tau[choose])-tau[choose]
    candidates={'actual_GPU_old_reference':arrays['proof_weights'],'old_reference_CPU':old_CPU,
        'native_GPU_weights':arrays['w'],'stable_padded_FP64_CPU':stable}
    errors={name:dict(max_abs=float(np.abs(value-serial).max()),RMSE=float(np.sqrt(np.mean((value-serial)**2))))for name,value in candidates.items()}
    ranked=np.argsort(np.abs(arrays['proof_weights']-serial))[-20:][::-1]
    records=[]
    for i in ranked:
        records.append(dict(sample_index=int(i),camera='DC',pixel_xy=arrays['pixel_xy'][ids[i]].tolist(),
            sigma=float(sigma[i]),starts=float(starts[i]),ends=float(ends[i]),tau=float(tau[i]),XYZ=arrays['xyz'][i].tolist(),
            native_GPU_weight=float(arrays['w'][i]),old_GPU_reference=float(arrays['proof_weights'][i]),old_CPU_reference=float(old_CPU[i]),
            serial_FP64=float(serial[i]),stable_padded_FP64=float(stable[i]),
            serial_exclusive_tau=float(local_prefix[i]),old_global_exclusive_tau=float(old_exclusive[i]),
            global_shift_minus_raybase_exclusive_tau=float(global_shift_subtract_base[i]),
            within_ray_cumsum_minus_current_tau=float(local_subtract_current[i]),
            global_cumsum_tau=float(global_cumsum[i]),global_cumsum_ulp=float(np.spacing(global_cumsum[i]))))
    sums={name:np.bincount(ids,weights=value,minlength=ray_count)for name,value in dict(candidates,serial_FP64=serial).items()}
    rgb={name:np.stack([np.bincount(ids,weights=value*arrays['rgb'][:,channel],minlength=ray_count)for channel in range(3)],-1)
        for name,value in dict(candidates,serial_FP64=serial).items()}
    composite={name:dict(alpha_max_error_vs_actual_native=float(np.abs(value-arrays['native_alpha'].reshape(-1)).max()),
        RGB_max_error_vs_actual_native=float(np.abs(rgb[name]-arrays['native_rgb']).max()))for name,value in sums.items()}
    np.savez_compressed(output/'reference_weight_diagnostics.npz',serial_FP64=serial,stable_FP64=stable,old_CPU=old_CPU,
        actual_old_GPU=arrays['proof_weights'],actual_native_GPU=arrays['w'],ids=ids)
    write(output/'diagnosis.json',dict(CPU_only=True,CUDA_initialized=torch.cuda.is_initialized(),optimizer_created=False,
        actual_witness_archive_sha256=sha(archive),actual_witness_completion_sha256=sha(completion_path),
        original_P2_observer_sha256=sha(observer),source_checkpoint_sha256=completion['source_checkpoint_sha256'],
        source_pipeline_digest=completion['source_pipeline_digest'],first_pixel=completion['first_pixel'],
        original_errors=completion['unchanged_observer_errors'],weight_errors_vs_serial=errors,composite_errors=composite,largest20_old_reference_errors=records,
        stable_matches_serial_1e12=bool(np.max(np.abs(stable-serial))<=1e-12),
        native_GPU_matches_serial_at_UNCHANGED2e5_tolerance=bool(np.max(np.abs(arrays['w']-serial))<=2e-5),
        reference_tolerance_unchanged=2e-5,stable_reference_accepted=False,automatic_recovery=False,
        interpretation='Actual saved CUDA native weights and original CUDA reference are compared against independent serial FP64 recurrence and padded exclusive FP64. These diagnostics do not change native field/sampler/compositor, precision, tolerance or authorize recovery. Root review remains required.'))
    print(json.dumps(errors))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--witness',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--observer',type=Path,required=True)
    args=parser.parse_args();run(args.witness,args.output,args.observer)
