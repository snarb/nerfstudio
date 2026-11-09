#!/usr/bin/env python3
"""Immutable global-exposure SDR/HDR/PQ variants from vendor-decoded raw TIFF.

preview builds three eval-view comparison artifacts only, never a fitting split.
build requires an explicit globally fixed gain, preserves every camera numeric
parameter, and exports SDRPNG, physical linearRec709EXR, archivalRec2020PQ16PNG
and clearly-labelled PQcodeEXR surrogate. No per-camera WB or exposure fitting.
"""
from __future__ import annotations
import argparse,importlib.util,json,hashlib,os,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2,numpy as np,tifffile
from PIL import Image,ImageDraw
PQ=Path('/fsx/oregon/gta_marshawn_2_1/six_sec_20261019/code/engine_gta_pv_export/ca4d/ca4d/pq.py')
RIG=Path('/fsx/oregon/gta_marshawn_2_1_prep_20261028/root/Marshawn_2_1/golden/rig')
RAW=Path('/fsx/oregon/gta_marshawn_2_1/cedar/tiff16/by_local')
VENDOR=Path('/fsx/oregon/gta_marshawn_2_1/cedar/prep/subpix/Marshawn_2_1/working/fullres/001660/images')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def vendor():
 spec=importlib.util.spec_from_file_location('marshawn_vendor_pq',PQ);pq=importlib.util.module_from_spec(spec);spec.loader.exec_module(pq);return pq

def linear_hd(row,pq):
 # Raw native lens pixels are decoded before spatial prefiltering, unlike old
 # display-resample pipeline. All exports derive from this one linear master.
 name=row['physical_camera'];path=RAW/name/(name+'.000408.tif');raw=tifffile.memmap(path,mode='r');linear=np.empty(raw.shape,np.float32);matrix=pq.rgb_to_rgb_matrix('rwg','rec709');negative=0
 for y in range(0,4320,128):
  camera=np.maximum(pq.log3g10_decode(raw[y:y+128].astype(float)/65535),0);value=camera@matrix.T;negative+=int((value<0).sum());linear[y:y+len(value)]=np.maximum(value,0).astype(np.float32)
 from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_text,read_images_text
 im=next(v for v in read_images_text(RIG/'images.txt').values() if v.name.startswith(name+'/'));cam=read_cameras_text(RIG/'cameras.txt')[im.camera_id];fx,fy,cx,cy,*d=cam.params;K=np.array([[fx,0,cx-.5],[0,fy,cy-.5],[0,0,1.]])
 w,h=row['w'],row['h'];target=np.array([[row['fl_x'],0,row['cx']-.5],[0,row['fl_y'],row['cy']-.5],[0,0,1.]])
 mx,my=cv2.initUndistortRectifyMap(K,np.array(d),None,target,(w,h),cv2.CV_32FC1);sx=w/8192;sy=h/4320;small=cv2.resize(linear,(w,h),interpolation=cv2.INTER_AREA);del linear
 image=cv2.remap(small,(mx+.5)*sx-.5,(my+.5)*sy-.5,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
 return image,dict(raw=str(path),raw_sha256=sha(path),negative_Rec709_fraction_before_positive_clip=negative/raw.size)
def srgb(linear):
 v=np.maximum(linear,0);v=v/(1+v);return np.rint(np.clip(np.where(v<=.0031308,12.92*v,1.055*v**(1/2.4)-.055),0,1)*255).astype(np.uint8)
def exr(path,rgb):
 if not cv2.imwrite(str(path),np.ascontiguousarray(rgb[...,::-1],dtype=np.float32),[cv2.IMWRITE_EXR_TYPE,cv2.IMWRITE_EXR_TYPE_FLOAT,cv2.IMWRITE_EXR_COMPRESSION,cv2.IMWRITE_EXR_COMPRESSION_ZIP]):raise OSError(path)
 check=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)[...,::-1]
 if not np.array_equal(check,np.asarray(rgb,np.float32)):raise ValueError('EXRfloat32 lossless roundtrip failed')
def pq_rgb(master,pq):
 rgb2020=np.maximum(master.astype(float)@pq.rgb_to_rgb_matrix('rec709','rec2020').T,0);nits=rgb2020*203;code=pq.pq_encode(np.clip(nits,0,10000)).astype(np.float32)
 return code,float((nits>10000).mean())
def preview(a):
 cv2.setNumThreads(2);a.output.mkdir(parents=True,exist_ok=False);m=json.loads((a.data/'transforms.json').read_text());pq=vendor();gains=[1.,4.,a.logmean_gain];rows=[next(r for r in m['frames'] if r['physical_camera']==name) for name in ['HC','DC','KA']];sheet=Image.new('RGB',(4*480,3*280));draw=ImageDraw.Draw(sheet);receipts=[]
 for i,row in enumerate(rows):
  linear,raw_receipt=linear_hd(row,pq);name=row['physical_camera'];panels=[('ExistingvendorJPEG (differentcanvas)',Image.open(VENDOR/(name+'_001660.jpg')).convert('RGB'))]
  for g in gains:
   image=srgb(linear*g);fn=f'{name}_gain{g:.6g}_sdr.png';Image.fromarray(image).save(a.output/fn);panels.append((f'{name} gain{g:.4g}, Reinhard+sRGB',Image.fromarray(image)))
  for j,(label,image) in enumerate(panels):image.thumbnail((480,253));sheet.paste(image,(j*480,i*280+24));draw.text((j*480+3,i*280+3),label,fill='white')
  receipts.append(dict(camera=name,raw=raw_receipt,linear_min=float(linear.min()),linear_max=float(linear.max())))
 sheet.save(a.output/'three_view_exposure_comparison.jpg',quality=93);write(a.output/'preview.json',dict(gains=gains,reference_white_nits=203,raw_records=receipts,training_started=False,canonical_data_changed=False,source_transforms_sha256=sha(a.data/'transforms.json'),source_pq_sha256=sha(PQ),script_sha256=sha(__file__)))
 print(a.output/'three_view_exposure_comparison.jpg',flush=True)
def build(a):
 cv2.setNumThreads(1);a.output.mkdir(parents=True,exist_ok=False);pq=vendor();meta=json.loads((a.data/'transforms.json').read_text());kinds={'sdr_png':'.png','linear_rec709_exr':'.exr','pq_rec2020_png16':'.png','pq_rec2020_codes_exr':'.exr'}
 for kind in kinds:
  (a.output/kind/'images').mkdir(parents=True);(a.output/kind/'masks').symlink_to((a.data/'masks').resolve(),target_is_directory=True)
 def one(row):
  linear,raw_receipt=linear_hd(row,pq);master=(linear*a.gain).astype(np.float32);code,clip_fraction=pq_rgb(master,pq);stem=Path(row['file_path']).stem;outputs={}
  for kind,ext in kinds.items():
   path=a.output/kind/'images'/(stem+ext)
   if kind=='sdr_png':Image.fromarray(srgb(master)).save(path)
   elif kind=='linear_rec709_exr':exr(path,master)
   elif kind=='pq_rec2020_codes_exr':exr(path,code)
   else:
    value=np.rint(code*65535).astype(np.uint16)
    if not cv2.imwrite(str(path),np.ascontiguousarray(value[...,::-1]),[cv2.IMWRITE_PNG_COMPRESSION,1]):raise OSError(path)
    if not np.array_equal(cv2.imread(str(path),cv2.IMREAD_UNCHANGED)[...,::-1],value):raise ValueError('PQPNG16 lossless roundtrip failed')
   outputs[kind]=dict(path=str(path),sha256=sha(path))
  print('converted',row['physical_camera'],flush=True);return dict(camera=row['physical_camera'],raw=raw_receipt,outputs=outputs,pq_10000nit_channel_clip_fraction=clip_fraction)
 with ThreadPoolExecutor(max_workers=a.workers) as pool:records=list(pool.map(one,meta['frames']))
 definitions={'sdr_png':dict(encoding='sRGB8display',primaries='Rec709/D65',transform='global exposure baked, Reinhard x/(1+x), sRGB OETF',native_loader='PNGuint8',native_cache='uint8'), 'linear_rec709_exr':dict(encoding='scene-linearfloat32',primaries='Rec709/D65',exposure_baked=True,nits_per_scene_unit=203.,display='Reinhard x/(1+x) then sRGB, no additional exposure',native_loader='EXRfloat32',native_cache='float32',physical_training='LookCloserHDR linear_softplus output, fixed203nits/unit PQobjective; compose linear before PQloss'), 'pq_rec2020_png16':dict(encoding='ST2084PQuint16',primaries='Rec2020/D65',nits_per_scene_unit=203.,peak_nits=10000.,native_loader='ARCHIVAL: nativePIL RGB PNGloader loses16bit precision; do nottrain directly'), 'pq_rec2020_codes_exr':dict(encoding='normalizedST2084PQcodefloat32',primaries='Rec2020/D65',nits_per_scene_unit=203.,peak_nits=10000.,native_loader='EXRfloat32 numericalpassthrough only',warning='CODE-SPACE SURROGATE ONLY; not scene-linear. Do not use as physical HDR input or implicit ns-eval sRGBdisplay')}
 for kind,ext in kinds.items():
  copied=json.loads(json.dumps(meta));remap={r['file_path']:'images/'+Path(r['file_path']).stem+ext for r in copied['frames']}
  for row in copied['frames']:row['file_path']=remap[row['file_path']]
  for split in ['train_filenames','val_filenames','test_filenames']:copied[split]=[remap[p] for p in copied[split]]
  write(a.output/kind/'transforms.json',copied);write(a.output/kind/'color_contract.json',dict(global_gain=a.gain,white_balance='unchanged as-shot; no perview WB',source_transforms_sha256=sha(a.data/'transforms.json'),**definitions[kind]))
 write(a.output/'complete.json',dict(images=len(records),train=63,eval=3,global_gain=a.gain,gain_stops=float(np.log2(a.gain)),source_transforms_sha256=sha(a.data/'transforms.json'),source_pq_sha256=sha(PQ),script_sha256=sha(__file__),data_derivation='Decode vendorLog3G10/RWG→positiveRec709linear, area-prefilter linear then one calibrated lens remap, one globally fixed exposure, export from commonmaster',poses_intrinsics_splits_preserved=True,white_balance='as-shot unchanged for allviews',reference_white_nits=203.,peak_nits=10000.,variants=definitions,records=records))
 print(json.dumps(dict(output=str(a.output),images=len(records),gain=a.gain)),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 for cmd in ['preview','build']:
  q=sub.add_parser(cmd);q.add_argument('--data',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
  if cmd=='preview':q.add_argument('--logmean-gain',type=float,required=True)
  else:q.add_argument('--gain',type=float,required=True);q.add_argument('--workers',type=int,default=2)
 a=p.parse_args()
 if a.command=='preview':preview(a)
 else:
  if not np.isfinite(a.gain) or a.gain<=0:p.error('Explicit finite positive global gain required')
  build(a)
