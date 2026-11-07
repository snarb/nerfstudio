"""Qualified243-ray auxiliary tau; private checkpointed RNG, no native sampler edits."""
import copy,hashlib,json
from pathlib import Path

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def validate_targets(path,expected_sha,data,support_manifest):
 import numpy as np
 path=Path(path);data=Path(data)
 if digest(path)!=expected_sha:raise ValueError('Qualified targets changed')
 s=json.loads(path.read_text());approved={'AA':(961,239),'BA':(954,175),'HA':(1067,113)}
 if len(s['targets'])!=243 or len(s['patches'])!=3:raise ValueError('Exactly three reviewed9x9 patches required')
 tf=json.loads((data/'transforms.json').read_text());frames={r['physical_camera']:r for r in tf['frames']if r['file_path']in tf['train_filenames']};box=np.array(json.loads((data/'ready.json').read_text())['aabb']);np.testing.assert_array_equal(box,np.asarray(s['AABB']));records={r['camera']:r for r in json.loads(Path(support_manifest).read_text())['records']};seen=set()
 for patch in s['patches']:
  ca=patch['camera'];x,y=approved[ca]
  if patch['center']!=[x,y]or patch['xyxy']!=[x-4,y-4,x+5,y+5]:raise ValueError('Unreviewed target polygon')
  record=records[ca];archive=Path(record['archive']);archive=path.parent/'source_archives'/f'{ca}.npz' if not archive.exists()else archive
  if record['archive_sha256']!=patch['source_archive_sha256']or digest(archive)!=patch['source_archive_sha256']or digest(data/'semantics'/f'{ca}.npz')!=patch['source_semantic_sha256']or digest(data/frames[ca]['file_path'])!=patch['source_RGB_sha256']:raise ValueError('Qualified patch input changed')
  with np.load(archive)as z:support=z['support'];unknown=z['unknown'];valid=z['valid']
  with np.load(data/'semantics'/f'{ca}.npz')as z:train_valid=z['valid'];w=z['W64']
  targets=[r for r in s['targets']if r['camera']==ca]
  if {(r['x'],r['y'])for r in targets}!={(xx,yy)for xx in range(x-4,x+5)for yy in range(y-4,y+5)}:raise ValueError('Duplicate/missing target pixels')
  frame=frames[ca];m=np.array(frame['transform_matrix']);o=m[:3,3]
  for r in targets:
   xx,yy=r['x'],r['y'];key=(ca,xx,yy)
   if key in seen:raise ValueError('Duplicate target');
   seen.add(key)
   if not(valid[yy,xx]and train_valid[yy,xx]and not support[yy,xx]and not unknown[yy,xx]and w[yy,xx]==0):raise ValueError('Target intersects actor/unknown/invalid')
   d=np.array([(xx+.5-frame['cx'])/frame['fl_x'],-(yy+.5-frame['cy'])/frame['fl_y'],-1.])@m[:3,:3].T;d/=np.linalg.norm(d);a=(box[0]-o)/d;b=(box[1]-o)/d;near=max(np.minimum(a,b).max(),.01);far=np.maximum(a,b).min();floor=-o[2]/d[2]
   if not(far>near and min(o[2]+near*d[2],o[2]+far*d[2])>.10 and floor>far):raise ValueError('Qualified floor/AABB geometry failed')
   for actual,expected in[(o,r['origin']),(d,r['direction']),([near,far,floor],[r['near'],r['far'],r['floor_t']])]:np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-9)
 return s


class QualifiedClearTau:
 def __init__(self,model,targets,sha256,coefficient):
  import torch
  if type(coefficient)is not float or coefficient not in(0.,.0001,.001):raise ValueError('Only reviewed0/.0001/.001 tau coefficients')
  self.model=model;self.coefficient=coefficient;self.target_sha256=sha256;self.generator=torch.Generator(device='cpu').manual_seed(73011);self.draws=0;self.first_hash=None;self.chain='0'*64;self.last=None
  self.o=torch.tensor([r['origin']for r in targets['targets']],device=model.device,dtype=torch.float32);self.d=torch.tensor([r['direction']for r in targets['targets']],device=model.device,dtype=torch.float32);self.near=torch.tensor([r['near']for r in targets['targets']],device=model.device,dtype=torch.float32);self.far=torch.tensor([r['far']for r in targets['targets']],device=model.device,dtype=torch.float32)
  if len(self.o)!=243:raise ValueError('Expected243qualified rays')
 def state_dict(self):return dict(schema=1,coefficient=self.coefficient,target_sha256=self.target_sha256,seed=73011,batch=64,samples=512,quadrature='independent stratified uniform jitter per step',draws=self.draws,first_hash=self.first_hash,chain=self.chain,generator_state=self.generator.get_state().clone())
 def load_state_dict(self,state):
  expected=self.state_dict()
  for k in('schema','coefficient','target_sha256','seed','batch','samples','quadrature'):
   if state[k]!=expected[k]:raise ValueError('Auxiliary source/recipe state differs: '+k)
  self.generator.set_state(state['generator_state'].cpu());self.draws=int(state['draws']);self.first_hash=state['first_hash'];self.chain=state['chain']
 def draw(self):
  import torch
  ids=torch.randint(243,(64,),generator=self.generator);jitter=torch.rand((64,512),generator=self.generator);raw=ids.numpy().tobytes()+jitter.numpy().tobytes();h=hashlib.sha256(raw).hexdigest();self.first_hash=self.first_hash or h;self.chain=hashlib.sha256(bytes.fromhex(self.chain)+raw).hexdigest();self.draws+=1
  return ids.to(self.o.device),jitter.to(self.o.device),h
 def tau(self,ids,jitter=None,samples=512):
  import torch
  from nerfstudio.cameras.rays import RaySamples,Frustums
  near,far=self.near[ids],self.far[ids];offset=.5 if jitter is None else jitter;t=near[:,None]+(far-near)[:,None]*(torch.arange(samples,device=near.device,dtype=near.dtype)[None]+offset)/samples;positions=(self.o[ids,None]+self.d[ids,None]*t[...,None]).reshape(-1,3);zeros=torch.zeros((len(positions),1),device=positions.device)
  rs=RaySamples(frustums=Frustums(origins=positions,directions=torch.zeros_like(positions),starts=zeros,ends=zeros,pixel_area=torch.ones_like(zeros)));sigma,_=self.model.field.get_density(rs);return sigma.float().reshape(len(ids),samples).mean(1)*(far-near)
 def loss(self):
  import torch
  ids,jitter,h=self.draw();tau=self.tau(ids,jitter)
  if not bool(torch.isfinite(tau).all())or bool((tau<0).any()):raise FloatingPointError('Nonfinite/negative qualifiedtau')
  self.last=dict(draw=self.draws,ray_jitter_sha256=h,mean_tau=float(tau.detach().mean()),max_tau=float(tau.detach().max()),coefficient=self.coefficient)
  return self.coefficient*tau.mean()
 def evaluate(self,pipe,targets,output,step):
  import torch
  from nerfstudio.engine.trainer import _capture_rng_state,_restore_rng_state
  rng=_capture_rng_state();training=self.model.training;self.model.eval();rows=[];order=[Path(p).stem.split('_')[-1]for p in pipe.datamanager.train_dataset.image_filenames];cameras=pipe.datamanager.train_dataset.cameras.to(self.model.device)
  try:
   with torch.no_grad():
    for ca in('AA','BA','HA'):
     ids=[i for i,r in enumerate(targets['targets'])if r['camera']==ca];q=[targets['targets'][i]for i in ids];camera=order.index(ca);coords=torch.tensor([[r['y']+.5,r['x']+.5]for r in q],device=self.model.device);rays=cameras[camera:camera+1].generate_rays(0,coords=coords,keep_shape=False);out=self.model(rays);tau=self.tau(torch.tensor(ids,device=self.o.device));rows.append(dict(camera=ca,targets=81,actual_native_alpha=out['accumulation'].flatten().cpu().tolist(),tau512_fixed_midpoint=tau.cpu().tolist(),mean_native_alpha=float(out['accumulation'].mean()),mean_tau512=float(tau.mean()),max_native_alpha=float(out['accumulation'].max())))
  finally:self.model.train(training);_restore_rng_state(rng)
  result=dict(step=step,source_sha256=self.target_sha256,rows=rows,auxiliary_generator_advanced=False,scope='Only243qualifiedpixels, no dense or unseenfree-space claim',evaluation_quadrature='fixedmidpoint512, not stochastic training jitter');p=Path(output)/'clear_targets'/f'step-{step:09d}.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n');return result
