"""Fresh63 TRAIN-only image-mask admission. No prior grid, heldout mask/map or neural state read.
Fixed floor AABB/calibration retained with disclosed legacy provenance.
"""
from pathlib import Path
import os,json,hashlib,time
import numpy as np
from scipy.ndimage import maximum_filter
from mask_visual_hull_aabb import MaskCamera,classify_cells
from build_marshawn_semantic_resolution import cell_bounds
from marshawn_semantic_cell_admission import WholeCellAdmission
from marshawn_video_support import load_support
KIND='PURE_MASK_SOURCE_GEOMETRY_BUNDLE_V1'
HELDOUT={'DC','HC','KA'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,j):Path(p).write_text(json.dumps(j,indent=2,allow_nan=False)+'\n')
def link(src,dest):dest.parent.mkdir(parents=True,exist_ok=True);os.link(Path(src).resolve(strict=True),dest)
def replay(data,chunk=32768):
 data=Path(data);tf=read(data/'transforms.json');box=read(data/'box.json');inputs=read(data/'admission_inputs/manifest.json');bounds=np.asarray(box['aabb']);frames={f['physical_camera']:f for f in tf['frames']if f['file_path']in tf['train_filenames']};rows={r['camera']:r for r in inputs['records']}
 if len(frames)!=63 or set(rows)!=set(frames) or set(frames)&HELDOUT:raise ValueError('Strict63 train-only calibrated masks required')
 bg=np.zeros(128**3,np.uint8);records=[];started=time.monotonic()
 for name in sorted(frames):
  row=rows[name];archive=data/row['archive']
  if sha(archive)!=row['archive_sha256']:raise ValueError('Mask INPUT changed '+name)
  with np.load(archive,allow_pickle=False)as z:
   if set(z.files)!={'support','unknown','valid'}:raise ValueError('Only Boolean image labels allowed')
   a={k:z[k]for k in z.files}
  f=frames[name]
  if any(v.dtype!=bool or v.shape!=(f['h'],f['w'])for v in a.values()):raise ValueError('Image label/camera raster differs')
  c=MaskCamera(name,np.asarray(f['transform_matrix']),f['fl_x'],f['fl_y'],f['cx'],f['cy'],a['support'],a['unknown'],a['valid']).prepare(0)
  for offset in range(0,len(bg),chunk):
   ids=np.arange(offset,min(offset+chunk,len(bg)));ids=ids[bg[ids]<2]
   if not len(ids):continue
   lo,hi=cell_bounds(ids,128,bounds);_,_,clear,_=classify_cells([c],lo,hi);bg[ids]+=clear.astype(np.uint8)
  records.append(dict(camera=name,archive_sha256=row['archive_sha256'],support_pixels=int(a['support'].sum()),unknown_pixels=int(a['unknown'].sum()),valid_pixels=int(a['valid'].sum())))
 base=(bg<2).reshape((128,)*3);pad=maximum_filter(base,size=3,mode='constant',cval=False);mapped=WholeCellAdmission(pad,bounds).map_target_grid((128,)*3,bounds)
 return dict(semantic_gate0=base,semantic_gate_pad1=pad,aabb=bounds,world_to_box=np.asarray(box['world_to_box'])),records,dict(seconds=time.monotonic()-started,mapped_bool_sha256=hashlib.sha256(mapped.tobytes()).hexdigest(),base=int(base.sum()),pad1=int(pad.sum()),mapped=int(mapped.sum()))

def build(source,output):
 source,out=Path(source),Path(output)
 if out.exists():raise ValueError('Preserve existing geometry')
 out.mkdir(parents=True);tf=read(source/'transforms.json');tf['frames']=[f for f in tf['frames']if f['physical_camera']not in HELDOUT];tf['train_filenames']=[f['file_path']for f in tf['frames']];tf['val_filenames']=[];tf['test_filenames']=[]
 if len(tf['frames'])!=63:raise ValueError('Wrong train camera split')
 write(out/'transforms.json',tf);link(source/'box.json',out/'box.json');old=read(source/'admission_inputs/manifest.json');records=[]
 for r in old['records']:
  if r['camera']in HELDOUT:continue
  link(source/r['archive'],out/r['archive']);records.append(r)
 inputs=dict(kind='STRICT63_BOOLEAN_MASK_ADMISSION_INPUTS_V1',frame=1660,train_camera_count=63,heldout_excluded=sorted(HELDOUT),native_pixel_support_width=64,records=records,no_frequency_or_model_read=True,origin_manifest_sha256=sha(source/'admission_inputs/manifest.json'))
 write(out/'admission_inputs/manifest.json',inputs);box=read(out/'box.json')
 write(out/'ready.json',dict(kind='STRICT63_GEOMETRY_ONLY_V1',frame=1660,aabb=box['aabb'],world_to_box=box['world_to_box'],train=63,heldout_excluded=sorted(HELDOUT),legacy_floor_bounds_calibration_retained=True))
 write(out/'prepared_ready.json',dict(admission_inputs_manifest_sha256=sha(out/'admission_inputs/manifest.json')))
 arrays,counts,stats=replay(out);ad=out/'admission';ad.mkdir();np.savez_compressed(ad/'semantic_grid.npz',**arrays)
 receipt=dict(frame=1660,train_cameras=sorted(r['camera']for r in records),native_pixel_support_width=64,no_learned_state_loaded=True,source_ready_sha256=sha(out/'ready.json'),transforms_sha256=sha(out/'transforms.json'),grid_sha256=sha(ad/'semantic_grid.npz'),builder_sha256=sha(__file__),admission_inputs_manifest_sha256=sha(out/'admission_inputs/manifest.json'),records=counts,clear_evidence='two distinct TRAIN63 cameras classify complete projected cell clear; UNKNOWN neverclear',stats=stats,fresh_train63_mask_recomputation=True,heldout_masks_or_maps_read=False,legacy_floor_bounds_calibration_retained=True)
 write(ad/'grid_receipt.json',receipt);write(ad/'manifest.json',dict(kind='FRAME_LOCAL_TRAIN63_W64_PAD1_WHOLE_CELL_V1',train_camera_count=63,heldout_excluded=sorted(HELDOUT),files_sha256={n:sha(ad/n)for n in ['grid_receipt.json','semantic_grid.npz']},mapped_bool_sha256=stats['mapped_bool_sha256'],no_model_or_learned_grid_reuse=True));return receipt

def load_support(manifest_path,expected_sha,data_dir,bounds,resolution,identity=False):
 path,data=Path(manifest_path),Path(data_dir)
 if sha(path)!=expected_sha:raise ValueError('Train63 admission manifest changed')
 m=read(path)
 if m['kind']!='FRAME_LOCAL_TRAIN63_W64_PAD1_WHOLE_CELL_V1'or m['train_camera_count']!=63 or set(m['heldout_excluded'])!=HELDOUT:raise ValueError('Strict63 admission required')
 for n,h in m['files_sha256'].items():
  p=Path(n)
  if p.is_absolute()or'..'in p.parts or sha(path.parent/p)!=h:raise ValueError('Admission artifact changed')
 r=read(path.parent/'grid_receipt.json');tf=read(data/'transforms.json');ready=read(data/'ready.json');names=sorted(f['physical_camera']for f in tf['frames']if f['file_path']in tf['train_filenames'])
 if len(names)!=63 or len(set(names))!=63 or set(names)&HELDOUT or names!=r['train_cameras']:raise ValueError('Admission camera leakage')
 if r['source_ready_sha256']!=sha(data/'ready.json')or r['transforms_sha256']!=sha(data/'transforms.json')or r['admission_inputs_manifest_sha256']!=sha(data/'admission_inputs/manifest.json'):raise ValueError('Admission provenance changed')
 inputs=read(data/'admission_inputs/manifest.json')
 if sorted(x['camera']for x in inputs['records'])!=names:raise ValueError('Admission inputs nottrain63')
 for x in inputs['records']:
  p=Path(x['archive'])
  if p.is_absolute()or'..'in p.parts or sha(data/p)!=x['archive_sha256']:raise ValueError('Train63 mask archive changed')
 if not r['fresh_train63_mask_recomputation']or r['heldout_masks_or_maps_read']or not r['no_learned_state_loaded']:raise ValueError('Invalid admission derivation')
 with np.load(path.parent/'semantic_grid.npz',allow_pickle=False)as z:base=z['semantic_gate0'].copy();source=z['semantic_gate_pad1'].copy();box=z['aabb'].copy();w2b=z['world_to_box'].copy()
 if base.dtype!=bool or source.dtype!=bool or source.shape!=(128,)*3 or not np.array_equal(source,maximum_filter(base,size=3,mode='constant',cval=False)):raise ValueError('Exactlyone halo required')
 np.testing.assert_allclose(box,ready['aabb'],rtol=0,atol=1e-12);np.testing.assert_allclose(w2b,ready['world_to_box'],rtol=0,atol=1e-12);np.testing.assert_allclose(bounds,box,rtol=0,atol=2e-7)
 if tuple(resolution)!=(128,)*3:raise ValueError('128cubed required')
 mapped=WholeCellAdmission(source,box).map_target_grid(resolution,bounds);digest=hashlib.sha256(mapped.tobytes()).hexdigest()
 if digest!=m['mapped_bool_sha256']:raise ValueError('Mapped admission changed')
 if identity:mapped=np.ones_like(mapped)
 return mapped,dict(manifest_sha256=expected_sha,source_grid_sha256=r['grid_sha256'],mapped_bool_sha256=hashlib.sha256(mapped.tobytes()).hexdigest(),frame=1660,train_camera_count=63,monitoring='DC/HC/KA heldout, never admission',identity_control=identity)
