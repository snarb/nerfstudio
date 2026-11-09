#!/usr/bin/env python3
"""Read-only ID/raster/value audit and log2 overlays for Marshawn HD frequencies.

Maps use an explicit patch lattice; uncovered raster tails are marked magenta,
not filled by resizing the map. Missing maps remain explicit while fitting runs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import html
import numpy as np
import torch
from PIL import Image,ImageDraw


def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()


def validate_recipe(meta):
 errors=[]
 threshold=meta.get('ssim_threshold')
 if not isinstance(threshold,(float,int)) or isinstance(threshold,bool) or not np.isfinite(threshold) or abs(threshold-.95)>1e-12:errors.append('recipe requires explicit ssim_threshold=0.95')
 if meta.get('train_steps_per_level')!=1000:errors.append('recipe requires explicit train_steps_per_level=1000')
 chunk=meta.get('eval_patch_batch_size')
 if not isinstance(chunk,int) or isinstance(chunk,bool) or chunk<=0:errors.append('recipe requires explicit positive eval_patch_batch_size')
 return errors

def colors(freq):
 t=np.clip((np.log2(np.maximum(freq,16))-4)/9,0,1)
 stops=np.array([[12,15,70],[0,170,210],[235,220,50],[245,35,20]])
 return np.stack([np.interp(t,[0,1/3,2/3,1],stops[:,c]) for c in range(3)],axis=-1).astype(np.uint8)

def inspect(row,data,directory):
 stem=Path(row['file_path']).stem;pt=directory/(stem+'.pt');jp=directory/(stem+'.json');errors=[];warnings=[]
 meta=json.loads(jp.read_text());recipe_errors=validate_recipe(meta);before=sha(pt);values=torch.load(pt,map_location='cpu',weights_only=True)
 if not torch.is_tensor(values):raise ValueError('Map must contain one scalar-resolution tensor')
 values=values.numpy();h,w=row['h'],row['w'];patch=int(meta.get('patch_size',0));stride=int(meta.get('stride',0))
 if patch<=0 or stride<=0:raise ValueError('Positive patch size/stride required')
 expected=((h-patch)//stride+1,(w-patch)//stride+1)
 checks=[(meta.get('image')==Path(row['file_path']).name,'image ID'),(meta.get('image_shape')==[h,w],'image raster'),(meta.get('crop_coords_xywh') is None,'unexpected crop'),(meta.get('value_type')=='scalar_resolution','value type'),(meta.get('min_res')==16,'min_res'),(meta.get('max_res')==8192,'max_res'),(meta.get('n_levels')==16,'n_levels'),(meta.get('n_features')==2,'n_features'),(meta.get('log2_hashmap_size')==23,'log2_hashmap_size'),(patch==8 and stride==8,'patch/stride8'),(values.shape==expected,'patch lattice shape'),(np.isfinite(values).all(),'finite values')]
 errors.extend(label for valid,label in checks if not valid)
 if values.ndim!=2 or not np.isfinite(values).all():raise ValueError('Invalid map rank/nonfinite values')
 if values.min()<16-1e-4 or values.max()>8192+1e-2:errors.append('values outside16..8192')
 schedule=16*(8192/16)**(np.arange(16)/15)
 if not np.allclose(meta.get('level_resolution_schedule',[]),schedule,rtol=1e-6):errors.append('level schedule')
 level=np.clip(np.rint(np.log(np.maximum(values,1e-10)/16)/np.log((8192/16)**(1/15))),0,15).astype(int);fractions=np.bincount(level.ravel(),minlength=16)/values.size
 if np.max(np.abs(values-schedule[level])/schedule[level])>1e-4:errors.append('values not on scalar-resolution level schedule')
 if fractions.max()>.98:warnings.append('More than98% patches in one level; inspect possible collapse')
 image=Image.open(data/row['file_path']).convert('RGB');mask=np.array(Image.open(data/row['mask_path']))>0
 if image.size!=(w,h) or mask.shape!=(h,w):raise ValueError('Source PNG or mask raster differs from transforms')
 covered_h=(values.shape[0]-1)*stride+patch;covered_w=(values.shape[1]-1)*stride+patch
 covered=np.zeros((h,w),bool);covered[:min(h,covered_h),:min(w,covered_w)]=True
 tail=int((~covered&mask).sum())
 if tail:warnings.append(f'{tail} valid pixels outside full-patch lattice; FAS patch draws alone cannot reach them')
 # Here stride=patch is validated; never stretch low-res map across trailing rows.
 expanded=np.repeat(np.repeat(colors(values),stride,axis=0),stride,axis=1)
 rgb=np.array(image);overlay=rgb.copy();eh=min(h,expanded.shape[0]);ew=min(w,expanded.shape[1]);overlay[:eh,:ew]=np.rint(.45*rgb[:eh,:ew]+.55*expanded[:eh,:ew]).astype(np.uint8)
 overlay[~covered&mask]=[255,0,255];overlay[~mask]=[60,60,60]
 if sha(pt)!=before:errors.append('map changed during review; rerun')
 receipt=data/'receipts'/(row['physical_camera']+'.json');source_hash=sha(data/row['file_path'])
 declared_source=meta.get('source_image_sha256')
 if declared_source is not None:
  if declared_source!=source_hash:recipe_errors.append('fit input source_image_sha256 mismatch')
 elif stem in {'train_AA','train_AC','train_BA','train_BC'} and meta.get('threshold_provenance','').startswith('original local run_lookcloser_preprocess_quiet command'):
  warnings.append('Retained original0.95 map uses historical command provenance instead of fit-time source hash')
 else:recipe_errors.append('missing fit-time source_image_sha256 or qualified original-map provenance')
 alignment_contract_valid=not errors;errors.extend(recipe_errors)
 if receipt.exists():
  old=json.loads(receipt.read_text())
  if old.get('output_sha256')!=source_hash:errors.append('prepared source receipt SHA mismatch')
 record=dict(camera=row['physical_camera'],image=row['file_path'],map=str(pt),metadata=str(jp),map_sha256=before,metadata_sha256=sha(jp),source_sha256=source_hash,mask_sha256=sha(data/row['mask_path']),map_shape=list(values.shape),image_shape=[h,w],covered_shape=[covered_h,covered_w],tail_valid_pixels=tail,min=float(values.min()),max=float(values.max()),median=float(np.median(values)),mean=float(values.mean()),nonempty_levels=int((fractions>0).sum()),perlevel_fractions=fractions.tolist(),ssim_threshold=meta.get('ssim_threshold'),train_steps_per_level=meta.get('train_steps_per_level'),eval_patch_batch_size=meta.get('eval_patch_batch_size'),fit_source_image_sha256=declared_source,threshold_provenance=meta.get('threshold_provenance'),alignment_contract_valid=alignment_contract_valid,recipe_errors=recipe_errors,recipe_valid=not recipe_errors,errors=errors,warnings=warnings,valid=not errors)
 return record,image,Image.fromarray(overlay)

def run(args):
 meta=json.loads((args.data/'transforms.json').read_text());directory=args.data/args.frequency_map_dir;train=set(meta['train_filenames']);rows=[r for r in meta['frames'] if r['file_path'] in train];expected={Path(r['file_path']).stem for r in rows}
 args.output.mkdir(parents=True,exist_ok=True);records=[];panels=[];missing=[]
 for row in rows:
  stem=Path(row['file_path']).stem;absent=[suffix for suffix in ['.pt','.json'] if not (directory/(stem+suffix)).exists()]
  if absent:missing.append(dict(camera=row['physical_camera'],missing=absent));continue
  try:
   record,source,overlay=inspect(row,args.data,directory);records.append(record)
   tile=Image.new('RGB',(640,190),'#101010');d=ImageDraw.Draw(tile)
   for x,im in [(0,source),(320,overlay)]:im.thumbnail((320,169));tile.paste(im,(x,21))
   d.text((3,3),f'{row["physical_camera"]} | source / log2 overlay | '+('VALID' if record['valid'] else 'INVALID'),fill='white');panels.append(tile)
  except Exception as exc:records.append(dict(camera=row['physical_camera'],valid=False,errors=[str(exc)]))
 pages=[]
 for start in range(0,len(panels),21):
  canvas=Image.new('RGB',(1920,1330),'#101010')
  for j,tile in enumerate(panels[start:start+21]):canvas.paste(tile,(j%3*640,j//3*190))
  name=f'frequency_contacts_{start//21+1:02d}.jpg';canvas.save(args.output/name,quality=90);pages.append(name)
 unexpected=[p.name for p in directory.glob('*.pt') if p.stem not in expected]
 dataset_errors=[]
 if len(rows)!=63 or len(expected)!=63:dataset_errors.append('Expected63distinct train image stems')
 if len(meta.get('val_filenames',[]))!=3 or train.intersection(meta.get('val_filenames',[])):dataset_errors.append('Expected3disjoint eval image paths')
 report=dict(dataset_errors=dataset_errors,recipe_contract=dict(ssim_threshold=.95,train_steps_per_level=1000,eval_patch_batch_size='explicit positive integer',fit_source_binding='source_image_sha256; only four documented original maps may use historical command provenance'),expected_maps=len(rows),complete_pairs=len(records),valid_maps=sum(r['valid'] for r in records),invalid_maps=[r for r in records if not r['valid']],incomplete=missing,unexpected_maps=unexpected,records=records,pages=pages,color_scale='log2 scalar resolution4..13; blue16 -> cyan -> yellow -> red8192; magenta=unrepresented valid raster tail; grey=invalid image mask',transforms_sha256=sha(args.data/'transforms.json'),script_sha256=sha(Path(__file__)),complete_gate=len(records)==len(rows) and all(r['valid'] for r in records) and not unexpected and not dataset_errors,caveats=['This validates files, coordinate contracts and visual map alignment; it does not prove downstream frequency effectiveness.','Patch-lattice tails require whole-image sampling or an explicit verified sampler extension.','Recipe release requires explicit0.95threshold and1000steps metadata; alignment-only historical QA is not recipe qualification.'])
 (args.output/'review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>HD frequency map review</h1><p>Recipe gate: '+str(report['complete_gate'])+'; '+str(report['valid_maps'])+'/63 valid. Requires explicit SSIM0.95 and1000steps/level. Historical alignment-only QA does not qualify the recipe.</p><p>'+html.escape(report['color_scale'])+'</p><a href="review.json">Statistics, hashes, incomplete maps and warnings</a>'+''.join(f'<a href="{n}"><img style="max-width:100%" src="{n}" loading="lazy"></a>' for n in pages))
 print(json.dumps({k:report[k] for k in ['expected_maps','complete_pairs','valid_maps','incomplete','unexpected_maps','complete_gate']}))
 return report['complete_gate']

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--frequency-map-dir',default='lookcloser_frequencies');p.add_argument('--output',type=Path,required=True);p.add_argument('--require-complete',action='store_true',help='Exit1unlessall63maps pass structure/source/recipe gates');a=p.parse_args();passed=run(a);raise SystemExit(0 if passed or not a.require_complete else 1)
