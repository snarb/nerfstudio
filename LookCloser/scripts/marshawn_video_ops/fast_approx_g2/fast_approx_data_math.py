"""Build baseline-compatible HD RGB/semantics from independent native video masks."""
from pathlib import Path
import argparse,json,shutil
import cv2,numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from prepare_marshawn_tuning import sha,write
from marshawn_actor_coordinates import raw_mask_coverage_to_HD
from marshawn_actor_confidence import native_confidence

SOURCE=Path('/fsx/tmp/marshawn_001660_hd/color_variants/radial_k4_joint_scalar_DIAGNOSTIC/sdr_png')

def main(a):
 cv2.setNumThreads(2);froot=a.root/'frames'/f'{a.frame:06d}';data=froot/('data_smoke_partial' if a.diagnostic_partial else 'data');data.mkdir(parents=True,exist_ok=True)
 if (data/'prepared_ready.json').exists():return
 box=json.loads(a.box.read_text());T=np.asarray(box['world_to_box']);tf=json.loads((SOURCE/'transforms.json').read_text());records=[];bindings=[]
 if len(tf['frames'])!=66:raise ValueError('All66 source cameras required')
 if a.diagnostic_partial:
  tf['frames']=[r for r in tf['frames'] if r['physical_camera'] in a.cameras]
 for name in ['images','semantics','masks','lookcloser_frequencies','review']:(data/name).mkdir(exist_ok=True)
 for row in tf['frames']:
  camera=row['physical_camera'];source=a.root/'sources'/str(a.frame)/camera;receipt=json.loads((source/'receipt.json').read_text());native=froot/'native_masks'/f'{camera}.npz';nmeta=json.loads(native.with_suffix('.json').read_text())
  if nmeta['frame']!=a.frame or nmeta['archive_sha256']!=sha(native) or nmeta['source_receipt_sha256']!=sha(source/'receipt.json'):raise ValueError('Frame-local native-mask source changed '+camera)
  original=Path(receipt['image'])
  if sha(original)!=receipt['image_sha256']:raise ValueError('Frame RGB changed '+camera)
  with np.load(native,allow_pickle=False) as z:
   masks={k:z[k].copy() for k in ['black','grey']};veto=z['source_ineligible'].copy();unknown=z['unknown'].copy()
  if any(m.shape!=(4320,8192) for m in masks.values()):raise ValueError('Native RAW raster required')
  nominal=masks['black']|masks['grey'];yy,xx=np.where(nominal)
  if not len(xx):
   yy=np.asarray([0,4319]);xx=np.asarray([0,8191])
  # Compute native-distance falloff on a crop with explicit zero surroundings,
  # then project through exactly the baseline RAW AREA-prefilter+single-remap.
  x0=max(0,int(xx.min())-129);x1=min(8192,int(xx.max())+130);y0=max(0,int(yy.min())-129);y1=min(4320,int(yy.max())+130)
  weights={}
  for key,width in [('W32',32),('W64',64)]:
   cropped,_=native_confidence(nominal[y0:y1,x0:x1],veto[y0:y1,x0:x1],width);full=np.zeros((4320,8192),np.float32);full[y0:y1,x0:x1]=cropped;weights[key]=raw_mask_coverage_to_HD(full,row);del full
  vetoHD=raw_mask_coverage_to_HD(veto.astype('float32'),row)>1e-6;unknownHD=raw_mask_coverage_to_HD(unknown.astype('float32'),row)>1e-6
  valid=np.asarray(Image.open(SOURCE/row['mask_path']).convert('L'))>0
  actorHD={k:(raw_mask_coverage_to_HD(v.astype('float32'),row)>=.5)&~vetoHD&valid for k,v in masks.items()}
  nominalHD=(raw_mask_coverage_to_HD(nominal.astype('float32'),row)>=.5)&~vetoHD&valid
  for w in weights.values():w[~valid|vetoHD]=0
  core=binary_fill_holes(cv2.morphologyEx(nominalHD.astype('uint8'),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17))))&valid&~vetoHD
  outer=cv2.dilate(core.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(33,33)))>0;yy=np.where(core)[0]
  if len(yy):
   wide=cv2.dilate(core.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(65,65)))>0;cut=int(yy.min()+.75*(yy.max()-yy.min()+1));outer[:cut]|=wide[:cut]
  semantic=data/'semantics'/f'{camera}.npz';np.savez_compressed(semantic,**actorHD,**weights,core=core,band=outer&~core&valid&~vetoHD,valid=valid,nominal=nominalHD,unknown=unknownHD|~valid,source_ineligible=vetoHD)
  row['file_path']=f'images/train_{camera}.png';dest=data/row['file_path']
  if not dest.exists():dest.symlink_to(original.resolve())
  row['mask_path']=f'masks/{camera}.png';Image.fromarray(valid.astype('uint8')*255).save(data/row['mask_path']);row['transform_matrix']=(T@np.asarray(row['transform_matrix'])).tolist()
  crops={};visibility={};visibility_review={}
  reviewed_path=a.root/'visibility_review.json'
  reviewed=json.loads(reviewed_path.read_text()).get('records',[]) if reviewed_path.exists() else []
  for actor,mask in actorHD.items():
   y,x=np.where(mask)
   if not len(x):
    visibility[actor]='approx_unavailable';visibility_review[actor]=dict(approximate=True,reason='No current approximate actor mask; no reviewed-absence claim');crops[actor]=None;continue
   visibility[actor]='present'
   crops[actor]=[max(0,int(x.min())-32),max(0,int(y.min())-32),min(mask.shape[1],int(x.max())+33),min(mask.shape[0],int(y.max())+33)]
  records.append(dict(camera=camera,image=row['file_path'],semantic=str(semantic),semantic_sha256=sha(semantic),rgb_sha256=sha(original),crops=crops,visibility=visibility,visibility_review=visibility_review))
  bindings.append(dict(camera=camera,source_receipt_sha256=sha(source/'receipt.json'),native_mask_sha256=sha(native),native_mask_receipt_sha256=sha(native.with_suffix('.json')),raw_sha256=receipt['raw_sha256']))
  im=np.array(Image.open(original).convert('RGB'));im[~nominalHD]//=4;preview=Image.fromarray(im);preview.thumbnail((640,360));preview.save(data/'review'/f'{camera}.jpg');print(a.frame,camera,flush=True)
 tf['train_filenames']=[r['file_path'] for r in tf['frames']];tf['val_filenames']=[r['file_path'] for r in tf['frames'] if r['physical_camera'] in ['DC','HC','KA']];tf['test_filenames']=list(tf['val_filenames']);tf['orientation_override']='none';tf.pop('applied_transform',None);tf.pop('applied_scale',None);tf['monitoring_policy']='DC/HC/KA overlapTRAIN; in-sample'
 write(data/'transforms.json',tf);write(data/'box.json',box);write(froot/'source_mask_binding.json',dict(frame=a.frame,records=bindings,coordinate_policy='NativeRAWmasks->AREA+singlecalibratedremap; samefrozenRGBphotometry/calibration',no_NeRF_or_learned_grid_state=True))
 write(data/'ready.json',dict(frame=a.frame,train=len(records),diagnostic_partial=a.diagnostic_partial,validation=['DC','HC','KA'],monitoring='in-sample',aabb=box['aabb'],world_to_box=box['world_to_box'],source_confidence_sha256=sha(froot/'source_mask_binding.json'),transforms_sha256=sha(data/'transforms.json'),records=records,frequency_policy='ALL66freshsource-boundSSIM.95maps pending',sparse_depth='not supplied; no transferredgeometry',mask_policy='FAST_APPROX_MASK_SOURCE_V1',mask_accuracy='approximate; user accepts boundary/identity imperfections'))
 write(data/('partial_ready.json' if a.diagnostic_partial else 'prepared_ready.json'),dict(frame=a.frame,train=len(records),cameras=sorted(r['camera'] for r in records),source_mask_binding_sha256=sha(froot/'source_mask_binding.json'),no_weights_optimizer_or_learned_grid_transfer=True,frequency_pending=True,mask_policy='FAST_APPROX_MASK_SOURCE_V1'))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--frame',type=int,required=True);p.add_argument('--box',type=Path,required=True);p.add_argument('--diagnostic-partial',action='store_true');p.add_argument('--cameras',nargs='+',default=[]);main(p.parse_args())
