"""Optional TRAIN63 proposal axes; defaults delegate byte-for-byte to frozen body80.

Head is a per-actor bbox-top30 spatial proxy, never a heldout/face annotation.
20% whole-valid and 2/15 protective-band floors remain. Analytic E_q[W64]
uses actual rounded source counts, per-camera means, equal camera probability,
FAS level/cell probabilities and balanced nonempty actor head pools.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from marshawn_tuning_sampling import quotas
from native_crop_sampling_v2 import NativeCropSamplingAdapterV2,install_sampling as base_install
DEFAULTS=dict(fas_body_fraction=.5,fas_level_max_weight=3.,fas_start_step=4096,fas_ramp_steps=4096,head_fraction=0.)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def probabilities(step,*,fas_body_fraction=.5,head_fraction=0.,fas_start_step=4096,fas_ramp_steps=4096):
 phase=min(1.,max(0.,(step-fas_start_step)/fas_ramp_steps));head=head_fraction*phase;body=2/3-head;fas=fas_body_fraction*phase
 return np.asarray([body*(1-fas),body*fas,2/15,1/5,head],np.float64)
def nondefault(spec):return any(spec.get(k,v)!=v for k,v in DEFAULTS.items())
def validate(spec):
 import math
 for k in ('fas_body_fraction','head_fraction'):
  v=spec.get(k,DEFAULTS[k])
  if isinstance(v,bool)or not isinstance(v,(int,float))or not math.isfinite(v)or not 0<=v<=(1 if k=='fas_body_fraction'else .3):raise ValueError('Invalid proposal fraction '+k)
 v=spec.get('fas_level_max_weight',3.)
 if isinstance(v,bool)or not isinstance(v,(int,float))or not math.isfinite(v)or not 1<=v<=5:raise ValueError('FAS level ratio must lie[1,5]')
 for k in ('fas_start_step','fas_ramp_steps'):
  v=spec.get(k,DEFAULTS[k])
  if type(v)is not int or v<(1 if k=='fas_ramp_steps'else 0):raise ValueError('Invalid FAS schedule '+k)
 if spec.get('head_fraction',0):
  p=Path(spec.get('head_proxy_manifest',''))
  if not p.is_absolute()or sha(p)!=spec.get('head_proxy_manifest_sha256'):raise ValueError('Head proxy manifest/hash required')
  m=json.loads(p.read_text())
  if m['kind']not in ('TRAIN63_NATIVE_BBOX_TOP30_HEAD_PROXY_V1','TRAIN63_NATIVE_SAM3_HEAD_FACE_IMPORTANCE_V1')or len(m['train_cameras'])!=63 or len(set(m['train_cameras']))!=63 or set(m['train_cameras'])&{'DC','HC','KA'}:raise ValueError('StrictTRAIN63 head spatialproxy required')
 elif 'head_proxy_manifest'in spec or 'head_proxy_manifest_sha256'in spec:raise ValueError('Zero head fraction must not bind head data')
 return spec

class ExperimentalSampling(NativeCropSamplingAdapterV2):
 def __init__(self,pipeline,data,objective,sampling,width,spec):
  self.parameters={k:spec.get(k,v)for k,v in DEFAULTS.items()};self.head_sha=spec.get('head_proxy_manifest_sha256');validate(spec)
  super().__init__(pipeline,data,objective,sampling,width);self.source_hits=np.zeros(5,np.int64);self.head_pools=[];self.proposal_means=[]
  manifest=None
  if self.parameters['head_fraction']:
   manifest=json.loads(Path(spec['head_proxy_manifest']).read_text());root=Path(spec['head_proxy_manifest']).parent
   if set(manifest['train_cameras'])!={Path(p).stem for p in pipeline.datamanager.train_dataset.image_filenames}:raise ValueError('Head camera membership differs from actual TRAIN')
  for index,(path,row)in enumerate(zip(pipeline.datamanager.train_dataset.image_filenames,self.rows)):
   c=Path(path).stem
   if c in {'DC','HC','KA'}:raise ValueError('Heldout sampler data forbidden')
   if self.parameters['fas_level_max_weight']!=3.:
    present=np.asarray([bool(x)for x in row['fas']['cells']],np.float64);weights=present*np.linspace(1.,self.parameters['fas_level_max_weight'],16);row['fas']['probabilities']=weights/weights.sum()if weights.sum()else weights
   means=np.asarray(row['proposal_means'],np.float64).copy();fas=row['fas'];fm=[]
   for cells in fas['cells']:fm.append(float(np.mean([row['weight'].reshape(-1)[p].mean(dtype=np.float64)for p in cells]))if cells else 0.)
   means[1]=float(np.dot(fas['probabilities'],fm))if fas['probabilities'].sum()else means[0]
   pools=[]
   if manifest is not None:
    rec=manifest['records'][c]if isinstance(manifest['records'],dict)else next(r for r in manifest['records']if r['camera']==c)
    if rec['semantic_sha256']!=self.source_records[index]['mask_sha256']:raise ValueError('Head source semantic SHA differs '+c)
    name=rec.get('file',rec.get('archive',c+'.npz'));p=root/name;digest=rec.get('sha256',rec.get('archive_sha256'))
    if Path(name).is_absolute()or'..'in Path(name).parts:raise ValueError('Owned relative proxy archive required')
    if sha(p)!=digest:raise ValueError('Head proxy raster changed '+c)
    with np.load(p,allow_pickle=False)as z:
     for actor in ('black','grey'):
      mask=z[actor]
      if mask.dtype!=bool or mask.shape!=row['core'].shape or np.any(mask&~(row[actor]&row['valid'])):raise ValueError('Head spatialproxy/source mismatch')
      pool=np.flatnonzero(mask)
      if len(pool):pools.append(pool)
   if not pools:pools=[self._pool(row,'core')]
   head_mean=float(np.mean([row['weight'].reshape(-1)[p].mean(dtype=np.float64)for p in pools]));self.head_pools.append(pools);self.proposal_means.append(np.append(means,head_mean))
 def _body_draw(self,batch):
  n=int(self.sampler.num_rays_per_batch);p=probabilities(self.step,**{k:v for k,v in self.parameters.items()if k!='fas_level_max_weight'});counts=quotas(p,n);sources=np.repeat(np.arange(5),counts);np.random.shuffle(sources);camera=np.random.randint(len(self.rows),size=n);flat=np.empty(n,np.int64)
  normalizer=float(np.mean([np.dot(counts/n,m)for m in self.proposal_means]))
  for c,row in enumerate(self.rows):
   for kind,name in ((0,'core'),(2,'band'),(3,'valid')):
    take=np.flatnonzero((camera==c)&(sources==kind));pool=self._pool(row,name);flat[take]=pool[np.random.randint(len(pool),size=len(take))]
   take=np.flatnonzero((camera==c)&(sources==1));fas=row['fas'];prob=fas['probabilities']
   if not prob.sum():pool=self._pool(row,'core');flat[take]=pool[np.random.randint(len(pool),size=len(take))]
   else:
    levels=np.searchsorted(np.cumsum(prob),np.random.random(len(take))).clip(0,15)
    for level,cells in enumerate(fas['cells']):
     if not cells:continue
     selected=take[levels==level];cellids=np.random.randint(len(cells),size=len(selected))
     for index,cellid in zip(selected,cellids):pool=cells[cellid];flat[index]=pool[np.random.randint(len(pool))]
   take=np.flatnonzero((camera==c)&(sources==4));pools=self.head_pools[c];actors=np.random.randint(len(pools),size=len(take))
   for actor,pool in enumerate(pools):selected=take[actors==actor];flat[selected]=pool[np.random.randint(len(pool),size=len(selected))]
  widths=np.asarray([r['core'].shape[1]for r in self.rows],np.int64)[camera];indices=torch.from_numpy(np.stack([camera,flat//widths,flat%widths],-1));self.source_hits+=np.bincount(sources,minlength=5);self.last_counts=counts.tolist()
  return self.packed_gather(batch,indices,keep_full_image=self.sampler.config.keep_full_image),normalizer
 def state_dict(self):return dict(super().state_dict(),experimental_proposal=self.parameters,head_manifest_sha256=self.head_sha)
 def load_state_dict(self,state):
  if state.get('experimental_proposal')!=self.parameters or state.get('head_manifest_sha256')!=self.head_sha:raise ValueError('Experimental proposal resume differs')
  super().load_state_dict(state)
 def audit_state(self):
  result=super().audit_state();result.update(source_names=['core_uniform','person_FAS','protective_band','whole_valid_image','balanced_actor_bbox_top30_proxy'],experimental_proposal=self.parameters,head_manifest_sha256=self.head_sha,normalizer='analytic mean_c sum_k actualroundedcounts[k]/N * E[W64|camera,source]; uniformcameras; balancednonemptyactorheadpools')
  return result

def install_sampling(pipeline,data,objective='actor',sampling='body80',width=64,*,spec=None):
 spec=spec or {};validate(spec)
 if not nondefault(spec):return base_install(pipeline,data,objective,sampling,width)
 if hasattr(pipeline,'marshawn_sampling_adapter'):raise ValueError('Sampler already installed')
 adapter=ExperimentalSampling(pipeline,data,objective,sampling,width,spec);pipeline.marshawn_sampling_adapter=adapter;return adapter
