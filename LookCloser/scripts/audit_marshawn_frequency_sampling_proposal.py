"""Audit only the CPU frequency sampling proposal against all63 original TRAIN rasters."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from marshawn_frequency_sampling_proposal import FrequencyProposal

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(data,out):
 ready=json.loads((data/'ready.json').read_text());tf=json.loads((data/'transforms.json').read_text());frames=[x for x in tf['frames']if x['file_path']in tf['train_filenames']]
 records={r['camera']:r for r in ready['records']};pools=[];shapes=[];sources=[];raw=[]
 for frame in frames:
  camera=Path(frame['file_path']).stem.removeprefix('train_');path=data/'semantics'/f'{camera}.npz'
  if camera in ready['validation']or sha(path)!=records[camera]['semantic_sha256']:raise ValueError('TRAIN source differs')
  with np.load(path)as z:row={k:z[k].copy()for k in ['core','band','valid']}
  row['core']&=row['valid'];row['band']&=row['valid'];assert not(row['core']&row['band']).any()
  shapes.append(row['valid'].shape);pools.append({k:np.flatnonzero(v)for k,v in row.items()});raw.append(row)
  sources.append(dict(camera=camera,semantic_sha256=sha(path),counts={k:int(v.sum())for k,v in row.items()}))
 if len(frames)!=63:raise ValueError('Expected63TRAIN')
 p=FrequencyProposal(shapes,pools);torch.manual_seed(42);counts={'original_core':0,'original_band':0,'focused_core':0,'focused_band':0};hist=np.zeros(63,int);total=0;fallbacks=0
 for update in range(32):
  before=torch.get_rng_state();p.prepare(list(range(63)),2048,before)
  ids=torch.randint(63,(2048,));shape=torch.tensor(shapes)[ids];y=(torch.rand(2048)*shape[:,0].float()).long();x=(torch.rand(2048)*shape[:,1].float()).long();after=torch.get_rng_state()
  yy,xx,valid=p.hook(ids);assert torch.equal(after,torch.get_rng_state());keep=p.last['labels']==2
  assert torch.equal(y[keep],yy[keep])and torch.equal(x[keep],xx[keep]);fallbacks+=p.last['fallbacks'];total+=len(ids)
  for i,row in enumerate(raw):
   select=ids.numpy()==i;hist[i]+=int(select.sum())
   for name in ('core','band'):
    counts['original_'+name]+=int(row[name][y[select],x[select]].sum());counts['focused_'+name]+=int(row[name][yy[select],xx[select]].sum())
 if torch.cuda.is_initialized():raise ValueError('CPU proposal initialized CUDA')
 out.mkdir(exist_ok=False,parents=True);result=dict(passed=True,scope='CPU proposal only; no field query/grid update/training installation',updates=32,rays=total,quotas_per2048=p.last['quotas'],source_names=['core','protective_band','historical_uniform'],camera_histogram=dict(zip([r['camera']for r in sources],hist.tolist())),counts=counts,fallbacks=fallbacks,CPU_global_rng_exact=True,original_uniform_pixels_exact=True,CUDA_initialized=False,GPU_rng_parity_not_yet_proven=True,main_sampling_and_objective_unchanged=True,sources=sources,source_ready_sha256=sha(data/'ready.json'),source_transforms_sha256=sha(data/'transforms.json'),prototype_sha256=sha(Path(__file__).with_name('marshawn_frequency_sampling_proposal.py')),helper_sha256=sha(__file__),caveat='Original core/band inherited; known bag falsepositives remain. This samples pixels, not true-surface labels. TRAIN geometry reserves and EVAL pixels never used as targets.')
 (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');(out/'private_state_example.json').write_text(json.dumps(p.state_dict(),indent=2)+'\n');print(json.dumps({k:result[k]for k in ['passed','rays','counts','fallbacks','CPU_global_rng_exact']}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.data,a.output)
