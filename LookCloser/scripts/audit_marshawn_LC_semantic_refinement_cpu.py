"""Actual learned128 equivalence and input hashes; CPU with CUDA hidden only."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
import torch
from marshawn_frozen_semantic_head import rendered_classes, tensor_digest
from marshawn_LC_semantic_refinement_contract import construct_head, BASE_SHA, PIPELINE_SHA
from prepare_marshawn_frozen_semantic_cpu import sha,write

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);a=p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
torch.set_num_threads(2)
assert sha(a.base)==BASE_SHA
base=torch.load(a.base,map_location='cpu',weights_only=False)
specs=[json.loads((a.root/'specs'/(arm+'_technical32.json')).read_text())for arm in ['coarse128','residual256']]
manifest=json.loads((a.campaign/'frozen_LC_semantic_head_silver_alias_v1/head_input_contract.json').read_text())
heads=[construct_head(base,s,manifest['bounds'])for s in specs]
assert tensor_digest(heads[0])==base['head_digest']
assert torch.count_nonzero(heads[1].residual_logits)==0
assert list(dict(heads[1].named_parameters()))==['residual_logits']
assert not heads[1].coarse_logits.requires_grad
optimizers=[torch.optim.Adam(h.parameters(),lr=.03)for h in heads]
assert all(not o.state_dict()['state']for o in optimizers)
witnesses=json.loads((a.root/'CPU_witnesses109.json').read_text())['records'];assert len(witnesses)==109
xyz=torch.tensor([r['witness_xyz']for r in witnesses],dtype=torch.float32)
foot=torch.tensor([[.057945191860198975,.05855029821395874,.01855337619781494]])
with torch.no_grad():
    probs=[h(xyz)for h in heads];feet=[float(h(foot)[0])for h in heads]
assert torch.equal(*probs)and feet[0]==feet[1]
capturep=a.campaign/'frozen_LC_semantic_head_silver_alias_v1/capture_adapter_receipt.json'
capture=json.loads(capturep.read_text());assert capture['model_digest']==PIPELINE_SHA
remote='/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106'
def local(path):return Path(path.replace(remote,str(a.campaign)))
# Source archives retain their exact targets/native weights. UNKNOWN confidence
# overlays occur only in ephemeral arrays inside the GPU fitter, as before.
records=[]
for row in capture['records']:
    if not row['rays']:continue
    path=local(row['archive']);assert sha(path)==row['archive_sha256']
    with np.load(path,allow_pickle=False)as z:
        assert str(z['model_digest'].item())==PIPELINE_SHA
        points=torch.from_numpy(z['xyz']);weights=torch.from_numpy(z['weight']);ids=torch.from_numpy(z['ray_indices'])
        with torch.no_grad():
            pred=[h(points)for h in heads]
            assert torch.equal(*pred)
            classes=[rendered_classes(v,weights,ids,row['rays'])for v in pred]
            assert torch.equal(*classes)
        records.append(dict(camera=row['camera'],rays=row['rays'],samples=len(points),source_sha256=row['archive_sha256'],probabilities_bitwise_equal=True,native_classes_bitwise_equal=True))
    assert sha(path)==row['archive_sha256']
assert len(records)==60 and sum(r['rays']for r in records)==69888
assert sha(a.base)==BASE_SHA and not torch.cuda.is_initialized()
write(a.root/'CPU_initial_function_qualification.json',dict(CPU_only=True,CUDA_initialized=False,
    base_checkpoint_sha256=BASE_SHA,base_iteration=2064,new_stage_iteration=0,both_Adam_state_empty=True,
    explicit_reset_Adam=True,parameter_names=[list(dict(h.named_parameters()))for h in heads],
    witnesses109_equal=True,unique_XYZ=len(np.unique(xyz.numpy(),axis=0)),footpoint_probability=feet,
    records=records,total_native_samples=sum(r['samples']for r in records),all_native_probabilities_and_classes_bitwise_equal=True,
    base_and_source_arrays_posthash_unchanged=True,source_model_loaded=False,
    witness_records=[dict(**r,coarse_probability=float(probs[0][i]),residual_probability=float(probs[1][i]))for i,r in enumerate(witnesses)],
    caveat='CPU equality only; sparse observations do not certify empty space or finalphysical quality. Fresh2064 requires GPU technical acceptance.'))
print(json.dumps(dict(CPU_only=True,CUDA_initialized=False,observations=109,foot=feet,caches=len(records),native_samples=sum(r['samples']for r in records),all_initial_functions_equal=True)))
