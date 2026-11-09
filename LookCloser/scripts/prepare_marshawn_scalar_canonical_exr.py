#!/usr/bin/env python3
"""Existing-metadata scalar derivative of canonical AP1 EXRs; no gain fitting."""
import argparse,hashlib,json,os
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2,numpy as np
from PIL import Image,ImageDraw

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def display(x,matrix):
 x=np.maximum(np.asarray(x,np.float64)@matrix.T,0)*4;x=x/(1+x)
 return np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)
def png(x):return Image.fromarray(np.rint(np.clip(x,0,1)*255).astype('uint8'))
def main(a):
 a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(2);out=a.output/'linear_ap1_exr';(out/'images').mkdir(parents=True);(out/'masks').symlink_to((a.source/'masks').resolve(),target_is_directory=True)
 src=json.loads((a.source/'color_contract.json').read_text());old=json.loads(a.scalar_metadata.read_text());assert old['eval_rgb_used_to_fit_gains']is False and src['camera_scalar_corrections']is False and src['extra_global_gain']==1
 tf=json.loads((a.source/'transforms.json').read_text());by={r['camera']:r for r in old['rows']};train={r['physical_camera']for r in tf['frames']if r['file_path']in tf['train_filenames']};assert len(train)==63 and len(tf['frames'])==66 and set(by)=={r['physical_camera']for r in tf['frames']};matrix=np.array(src['ap1_to_rec709_preview_matrix']);records=[];luminance=[];preview=Image.new('RGB',(1440,840),'#111111');draw=ImageDraw.Draw(preview)
 for row in tf['frames']:
  n=row['physical_camera'];ev=float(by[n]['exposure_correction_ev']);source=a.source/row['file_path'];x=cv2.imread(str(source),cv2.IMREAD_UNCHANGED);assert x.dtype==np.float32 and x.shape==(1012,1920,3)and np.isfinite(x).all()and(x>=0).all();y=x*(2**ev);dest=out/row['file_path'];assert cv2.imwrite(str(dest),y,[cv2.IMWRITE_EXR_TYPE,cv2.IMWRITE_EXR_TYPE_FLOAT,cv2.IMWRITE_EXR_COMPRESSION,cv2.IMWRITE_EXR_COMPRESSION_ZIP]);check=cv2.imread(str(dest),cv2.IMREAD_UNCHANGED);assert np.array_equal(check,y)
  if'mask_path'in row:
   mask=np.asarray(Image.open(a.source/row['mask_path']));assert mask.shape==(1012,1920)and(mask>0).all()
  if n in train:luminance.append((y[...,2]*.2722287168+y[...,1]*.6740817658+y[...,0]*.0536895174).reshape(-1))
  records.append(dict(camera=n,split='train'if n in train else'eval',existing_EV=ev,source=str(source),source_sha256=sha(source),output=str(dest),output_sha256=sha(dest),FP32_exact_scalar_multiply=True,mask_valid=True,gain_source=by[n]['gain_source']))
  if n in['DC','HC','KA']:
   i=['DC','HC','KA'].index(n);rgb=x[...,::-1];newrgb=y[...,::-1];fixed=display(newrgb,matrix);reference=display(rgb*(2**ev),matrix);assert np.max(abs(fixed-reference))<1e-6;unscalar=display(newrgb/(2**ev),matrix)
   for j,(value,label)in enumerate([(reference,'Originalcanonical +existingEV DISPLAY'),(fixed,'DerivedEXR fixed4xONLY DISPLAY'),(unscalar,'DerivedEXR existingEV UNDONE DISPLAY')]):
    im=png(value);im.save(a.output/f'{n}_panel{j}_SOURCE_GT.png');im.thumbnail((480,253));preview.paste(im,(j*480,i*280+24));draw.text((j*480+3,i*280+5),n+' '+label,fill='white')
  print(n,'exact',ev,flush=True)
 (out/'transforms.json').write_bytes((a.source/'transforms.json').read_bytes());assert sha(out/'transforms.json')==sha(a.source/'transforms.json');ys=np.concatenate(luminance);q50,q999=np.quantile(ys,[.5,.999]);stats=dict(train_cameras=63,pixels=len(ys),luma='AP1D60 [.2722287168,.6740817658,.0536895174]',q50=float(q50),q999=float(q999),uses_eval_RGB=False)
 contract=dict(kind='DERIVED_existing_camera_scalar_calibrated_AP1_LINEAR_NOT_VENDOR_RAW',representation='scene-linearfloat32EXR',primaries='ACESAP1_D60',source_master=str(a.source),source_contract_sha256=sha(a.source/'color_contract.json'),source_transforms_sha256=sha(a.source/'transforms.json'),existing_scalar_metadata=str(a.scalar_metadata),existing_scalar_metadata_sha256=sha(a.scalar_metadata),operation='FLOAT32canonicalrelativeAP1 *2**EXISTINGcameraEV ONCE;no gain4baked; no newfit',camera_scalar_corrections=True,extra_global_gain=1.,global_gain4_baked=False,existing_per_camera_EV_baked=True,eval_rgb_used_to_fit_gains=False,old_eval_EV_source=old['eval_gain_predictions'],signed_gain_reference_nits_per_calibrated_linear_RGB_unit=52.13360318135631,signed_floor_nits=.005,signed_content_peak_nits=4000.,PQ_normalization_nits=10000.,radiometric_scope='Inheritedsignedtake referenceusedforCALIBRATEDDERIVATIVE;NOTbyteidentitywithvendorrawPQornewabsoluteradiometricmeasurement.',loss_target='PQ(clip(52.13360318135631*calibratedAP1,.005,4000))',ap1_to_rec709_preview_matrix=matrix.tolist(),common_display='AP1→Rec709D65→4xONLY→Reinhard+sRGB; DO NOTapplycameraEVagain',unscalar_display='UndoexistingcameraEVinlinearAP1 first, thenfixed4xcommontransform',white_balance='No newWB/RGBmatrix/vignettefit',geometry_splits_masks_unchanged=True,train_AP1_statistics=stats,rows=records,script_sha256=sha(__file__),quality_claim=False)
 (out/'color_contract.json').write_text(json.dumps(contract,indent=2)+'\n');preview.save(a.output/'all3_source_display_identity.jpg',quality=94);(a.output/'complete.json').write_text(json.dumps(contract,indent=2)+'\n');(a.output/'QA.json').write_text(json.dumps(dict(passed=True,images=66,train=63,eval=3,all66_FP32_scalar_bytes_exact=True,all66_masks_valid=True,transforms_byteidentical=True,no_double_EV_display=True,newfit=False,NN_training=False,root_visual_review_pending=True),indent=2)+'\n');print(json.dumps(stats))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('/fsx/tmp/marshawn_001660_hd/color_variants/canonical_signed_ap1/linear_ap1_exr'));p.add_argument('--scalar-metadata',type=Path,default=Path('/fsx/tmp/marshawn_001660_hd/color_variants/gain4_scalar_trainfit_poseeval/complete.json'));p.add_argument('--output',type=Path,required=True);main(p.parse_args())
