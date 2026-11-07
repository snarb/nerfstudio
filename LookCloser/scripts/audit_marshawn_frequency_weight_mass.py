"""CPU sample-position frequency lookup, preserving original native rendered weights."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def summarize(rows):
 opacity=sum(r['opacity']for r in rows);band=sum(r.get('witness_band_mass',0)for r in rows)
 return dict(rays=len(rows),mean_opacity=opacity/len(rows),frequency={str(level):dict(mean_original_ray_mass=sum(r['level_mass'][str(level)]for r in rows)/len(rows),fraction_of_termination=sum(r['level_mass'][str(level)]for r in rows)/opacity if opacity else None,mean_witness_band_mass=sum(r.get('level_band_mass',{}).get(str(level),0)for r in rows)/len(rows),fraction_of_witness_band_mass=sum(r.get('level_band_mass',{}).get(str(level),0)for r in rows)/band if band else None)for level in [0,15]})
def main(root,out):
 checkpoint=root/'checkpoint.ckpt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False);state=payload['pipeline'];keys=[k for k in state if k.endswith('.freq_grid.grid')];assert keys
 key=next((k for k in keys if k.endswith('_model.freq_grid.grid')),keys[0]);grid=state[key].cpu();prefix=key[:-4];lo=state[prefix+'aabb_min_buf'].cpu();size=state[prefix+'aabb_size_buf'].cpu();res=grid.shape[0]
 if not all(torch.equal(grid,state[k].cpu())for k in keys):raise ValueError('Duplicated frequency buffers disagree')
 if set(torch.unique(grid).tolist())!={0.,15.}:raise ValueError('Expected currently observed0/15 levels')
 output={}
 for label in ['clothing103','DC12']:
  folder=root/(label+'_trace/rays');audit=json.loads((folder/'audit.json').read_text());rows=[]
  if sha(checkpoint)!=audit['checkpoint_sha256']:raise ValueError('Trace/checkpoint mismatch')
  for ray in audit['rows']:
   path=folder/ray['profiles']['native']['trace']
   with np.load(path)as src:
    xyz=src['xyz'].copy();weight=src['actual_weights'].reshape(-1).astype(np.float64);mid=(src['t_start'].reshape(-1)+src['t_end'].reshape(-1))*.5
   positions=torch.from_numpy(xyz).float();coords=torch.clamp((positions-lo)/size*(res-1),0,res-1.001);idx=coords.long();levels=grid[idx[:,0],idx[:,1],idx[:,2]].numpy()
   reference=np.clip((xyz.astype(np.float32)-lo.numpy())/size.numpy()*(res-1),0,np.float32(res-1.001)).astype(np.int64)
   if not np.array_equal(idx.numpy(),reference):raise ValueError('Independent float32 frequency indexing differs')
   row=dict(pixel=ray['pixel'],opacity=float(weight.sum()),level_mass={str(v):float(weight[levels==v].sum())for v in [0,15]},samples_by_level={str(v):int((levels==v).sum())for v in [0,15]},trace_sha256=sha(path))
   if 'witness_xyz'in ray['pixel']:
    depth=float((np.array(ray['pixel']['witness_xyz'])-ray['origin'])@np.array(ray['direction']));band=abs(mid-depth)<=.02;row.update(witness_band_mass=float(weight[band].sum()),level_band_mass={str(v):float(weight[band&(levels==v)].sum())for v in [0,15]})
   if abs(sum(row['level_mass'].values())-row['opacity'])>1e-12:raise ValueError('Original mass partition failed')
   rows.append(row)
  groups={g:summarize([r for r in rows if r['pixel'].get('diagnostic_group')==g])for g in sorted({r['pixel'].get('diagnostic_group')for r in rows}-{None})}
  output[label]=dict(rows=rows,summary=summarize(rows),groups=groups)
 if torch.cuda.is_initialized():raise ValueError('CPU audit initialized CUDA')
 out.mkdir(parents=True,exist_ok=True);report=dict(kind='ORIGINAL_NATIVE_WEIGHT_MASS_BY_FREQUENCY',step=payload['step'],checkpoint_sha256=sha(checkpoint),frequency_buffer=key,frequency_grid_sha256=hashlib.sha256(grid.numpy().tobytes()).hexdigest(),frequency_bounds=[lo.tolist(),(lo+size).tolist()],resolution=res,profiles=output,CUDA_initialized=False,weight_source='actual_weights from native renderer, unrenormalized',band_policy='same midpoint ±.02 classification as sealed103 audit',caveats=['Frequency at refined sample position is not necessarily the frequency chosen at its parent coarse-interval midpoint.','Level0 does not mean physically empty; it denotes the stored minimum frequency.','TRAIN witnesses are source geometry with inherited uncertainty, not independent complete-scene truth.'])
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v['summary']for k,v in output.items()}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.source,a.output)
