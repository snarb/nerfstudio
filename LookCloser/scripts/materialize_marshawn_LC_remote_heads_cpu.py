"""Copy completed root-released LC heads without rewriting scientific artifacts.

Original remote JSON/checkpoints remain byte-identical. A separate completion
adapter records the local path used by later independently released rendering.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
from prepare_marshawn_frozen_semantic_cpu import sha,write


def fetch(host,path,target):
    with target.open('xb')as f:
        subprocess.run(['ssh',host,'cat -- '+shlex.quote(str(path))],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.PIPE,check=True)


def run(alias,output,host):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():raise ValueError('CPU-only materialization required')
    equivalence_path=alias/'canonical_equivalence.json';equivalence=json.loads(equivalence_path.read_text())
    if not all(equivalence[k]for k in ['scientific_numeric_objective_seed_prior_LR_grid_camera_packets_unchanged',
            'original63_membership_69888_rows_136_capacity_identities_two_UNKNOWN_retained',
            'all_cache_arrays_and_source_metadata_byte_identical','canonical_JSONs_NOT_rewritten']):
        raise ValueError('Explicit canonical equivalence required')
    remote_specs={Path(p).name:(Path(p),digest)for p,digest in equivalence['alias_specs_sha256'].items()}
    remote_root=next(iter(remote_specs.values()))[0].parent.parent
    if output.exists():raise FileExistsError('Use a fresh local artifact namespace')
    output.mkdir(parents=True)
    fetch(host,remote_root/'root_release.json',output/'original_remote_root_release.json')
    release=json.loads((output/'original_remote_root_release.json').read_text())
    if release.get('reviewer')!='root'or release.get('LC_first2064_head_authorized')is not True:
        raise ValueError('Actual root fresh2064 release required before accepting remote artifacts')
    records=[]
    for name,prior in [('prior099_fresh2064',.99),('prior001_fresh2064',.01)]:
        directory=output/name;directory.mkdir()
        spec_path,spec_digest=remote_specs[name+'.json']
        local_spec=alias/'specs'/(name+'.json')
        if sha(local_spec)!=spec_digest or release['spec_sha256'].get(str(spec_path))!=spec_digest:
            raise ValueError('Root alias spec binding differs')
        spec=json.loads(local_spec.read_text())
        for remote_name,local_name in [('complete.json','original_complete.json'),('initial_state.json','original_initial_state.json'),('head.pt','head.pt')]:
            fetch(host,Path(spec['output'])/remote_name,directory/local_name)
        completion=json.loads((directory/'original_complete.json').read_text())
        initial=json.loads((directory/'original_initial_state.json').read_text())
        checkpoint=directory/'head.pt'
        if (sha(checkpoint)!=completion['checkpoint_sha256']or completion['actual_head_updates']!=2064 or completion['iteration']!=2064 or
                initial['initial_iteration']!=0 or initial['initial_Adam_count']!=0 or initial['restoration']is not None or
                not completion['native_model_not_loaded_or_optimized']or not completion['native_checkpoint_and_caches_unchanged']):
            raise ValueError('Actual fresh2064 execution/provenance required')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        canonical_contract=json.loads((Path(equivalence['source_local_head'])/'head_input_contract.json').read_text())
        if (state['spec_sha256']!=spec_digest or state['iteration']!=2064 or state['source_kind']!='DIRECT_FROZEN_LC_NATIVE_VISIBILITY' or
                state['initial_probability']!=prior or state['model_digest']!=equivalence['source_pipeline_digest']or
                state['LC_capture_receipt_sha256']!=equivalence['canonical_capture_receipt_sha256']or
                state['label_qualification_sha256']!=equivalence['alias_label_qualification_sha256']or
                not state['world_to_box_is_identity']or not torch.equal(state['head']['world_to_box'],torch.eye(4))or
                state['head']['logits'].shape!=(1,1,128,128,128) or
                not torch.equal(state['head']['bounds'],torch.tensor(canonical_contract['bounds'],dtype=torch.float32)) or
                state['head']['logits'].dtype!=torch.float32 or not torch.isfinite(state['head']['logits']).all()):
            raise ValueError('Remote head scientific contract differs')
        reference=Path(equivalence['source_local_head']).parent/'frozen_native_semantic_prior001_candidate_v1/camera_history_reference.json'
        if state['camera_history']!=json.loads(reference.read_text())['camera_history']:
            raise ValueError('Actual2064 camera packets differ from both controlled priors')
        adapter=dict(completion,kind='DERIVED_LOCAL_PATH_COMPLETION_ADAPTER',checkpoint=str(checkpoint),
            original_completion_sha256=sha(directory/'original_complete.json'),original_remote_checkpoint=completion['checkpoint'],
            original_remote_release_sha256=sha(output/'original_remote_root_release.json'),
            canonical_equivalence_sha256=sha(equivalence_path),scientific_checkpoint_bytes_unchanged=True)
        write(directory/'complete.json',adapter)
        records.append(dict(name=name,prior=prior,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
            completion_adapter=str(directory/'complete.json'),completion_adapter_sha256=sha(directory/'complete.json'),
            original_completion_sha256=sha(directory/'original_complete.json'),remote_training_spec=str(spec_path),remote_training_spec_sha256=spec_digest,
            alias_label_qualification_sha256=state['label_qualification_sha256'],source_pipeline_digest=state['model_digest']))
    write(output/'materialization.json',dict(CPU_only=True,CUDA_initialized=torch.cuda.is_initialized(),optimizer_created=False,
        records=records,canonical_equivalence=str(equivalence_path),canonical_equivalence_sha256=sha(equivalence_path),
        remote_root_release_sha256=sha(output/'original_remote_root_release.json'),
        source_capture_receipt_sha256=equivalence['canonical_capture_receipt_sha256'],no_implicit_hash_exemptions=True,
        disclaimer='Local path adapters only. Does not authorize TRAIN audit/physical rendering, accept any selector or change source labels/geometry.'))
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--alias',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--host',default='ubuntu@silver-sable');args=parser.parse_args();run(args.alias,args.output,args.host)
