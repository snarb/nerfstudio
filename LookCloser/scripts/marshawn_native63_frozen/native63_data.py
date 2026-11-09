"""Strict63 training inventory. Heldout masks are evaluation-only; heldout maps forbidden."""
from pathlib import Path
import os,json,hashlib,copy
HELDOUT={'DC','HC','KA'}
KIND='NATIVE63_HELDOUT3_FREQUENCY_DATA_V1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def membership(tf):
 frames={r['file_path']:r for r in tf['frames']};train=tf['train_filenames'];ev=tf['val_filenames'];tc={frames[x]['physical_camera']for x in train};ec={frames[x]['physical_camera']for x in ev}
 if len(frames)!=66 or len(train)!=63 or len(set(train))!=63 or len(ev)!=3 or len(set(ev))!=3 or set(train)&set(ev)or set(train)|set(ev)!=set(frames)or tc&HELDOUT or ec!=HELDOUT:raise ValueError('Strict63/DC HC KA disjoint split required')
 return frames,tc,ec

def verify_data(data,expected_policy=None):
 import torch,numpy as np
 data=Path(data);r=read(data/'ready.json');tf=read(data/'transforms.json');frames,tc,ec=membership(tf)
 if r['kind']!=KIND or r['train']!=63 or set(r['validation'])!=HELDOUT or r['in_sample']is not False or r['transforms_sha256']!=sha(data/'transforms.json'):raise ValueError('Wrong strict63 ready')
 if expected_policy is not None and r['frequency_source_policy']!=expected_policy:raise ValueError('Frequency policy differs')
 for n,h in r['inventory']['files_sha256'].items():
  p=Path(n)
  if p.is_absolute()or'..'in p.parts or sha(data/p)!=h:raise ValueError('Strict63 artifact changed '+n)
 freq=data/'lookcloser_frequencies';expected={c+x for c in tc for x in ('.pt','.json')}
 if {p.name for p in freq.iterdir()if p.is_file()}!=expected:raise ValueError('Heldout/unlisted/missing frequency map')
 ladder=torch.tensor([16*512**(i/15)for i in range(16)],dtype=torch.float32)
 rows={x['camera']:x for x in r['records']}
 if len(rows)!=66 or set(rows)!={f['physical_camera']for f in frames.values()}:raise ValueError('Source metadata coverage changed')
 for p,f in frames.items():
  c=f['physical_camera'];row=rows[c]
  if row['rgb_sha256']!=sha(data/p)or row['semantic_sha256']!=sha(data/'semantics'/(c+'.npz')):raise ValueError('Native source changed')
  if c in ec:continue
  m=read(freq/(c+'.json'));v=torch.load(freq/(c+'.pt'),map_location='cpu',weights_only=True)
  if m.get('camera')!=c or m.get('source_image_sha256')!=row['rgb_sha256']or m.get('value_type')!='scalar_resolution'or m.get('patch_size')!=8 or m.get('stride',8)!=8 or m.get('image_shape')!=[f['h'],f['w']]or v.dtype!=torch.float32 or list(v.shape)!=[f['h']//8,f['w']//8]or not torch.isfinite(v).all():raise ValueError('Per-camera frequency provenance/raster differs '+c)
  lev=torch.round(torch.log(v/16)/torch.log(torch.tensor(512.))*15).long().clamp(0,15)
  if not torch.allclose(v,ladder[lev],rtol=2e-6,atol=1e-5):raise ValueError('Frequency off16-level scalar ladder')
  if r['frequency_source_policy']=='progressive_ssim':
   if m['ssim_threshold']!=.95 or m['train_steps_per_level']!=1000 or not m['progressive_fit_from_scratch']or m['previous_frame_state_loaded']:raise ValueError('NN frequency fit provenance differs')
  elif r['frequency_source_policy']in ('uniform16','gradient_rank'):
   if m.get('proxy_kind')!=r['frequency_source_policy']or m.get('train_steps_per_level')!=0 or m.get('ssim_threshold','missing')is not None or not m.get('estimator_no_training'):raise ValueError('Honest zero-fit proxy required')
  else:raise ValueError('Unsupported first-wave policy')
 return r

def build(base,output,policy,map_root=None):
 base,out=Path(base),Path(output)
 if out.exists():raise ValueError('Preserve all previous dataset generations')
 out.mkdir(parents=True);tf=read(base/'transforms_63train3eval.json');frames,tc,ec=membership(tf);old=read(base/'ready.json');records=copy.deepcopy(old['records']);files={}
 def link(src,name):
  p=out/name;p.parent.mkdir(parents=True,exist_ok=True);os.link(Path(src).resolve(),p);files[name]=sha(p)
 write(out/'transforms.json',tf);files['transforms.json']=sha(out/'transforms.json');link(base/'box.json','box.json')
 for r in records:
  c=r['camera'];r['semantic']=str(out/'semantics'/(c+'.npz'));r['split']='eval'if c in ec else'train'
  for n in (r['image'],'masks/'+c+'.png','semantics/'+c+'.npz','receipts/'+c+'.json'):link(base/n,n)
  if c in tc:
   maps=Path(map_root)if map_root else base/'lookcloser_frequencies'
   for ext in ('.pt','.json'):link(maps/(c+ext),'lookcloser_frequencies/'+c+ext)
 box=read(out/'box.json');ready=dict(kind=KIND,frame=1660,train=63,eval=3,validation=['DC','HC','KA'],train_camera_ids=sorted(tc),eval_camera_ids=sorted(ec),in_sample=False,records=records,aabb=box['aabb'],world_to_box=box['world_to_box'],transforms_sha256=sha(out/'transforms.json'),frequency_source_policy=policy,frequency_train_camera_count=63,heldout_frequency_maps_present=False,source_RGB_masks_camera_bytes_unchanged=True,frequency_maps_per_camera_independent_reuse=True,no_model_optimizer_or_learned_grid_loaded=True,legacy_floor_bounds_calibration_retained=True,inventory=dict(files_sha256=files),builder_sha256=sha(__file__))
 write(out/'ready.json',ready);return verify_data(out,policy)
