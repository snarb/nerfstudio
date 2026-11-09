#!/usr/bin/env python3
"""Read-only numerical replay of the experimental scalar exposure dataset."""
import argparse,hashlib,json,os
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2,numpy as np
from PIL import Image

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def encode(v):
 x=np.maximum(v,0);x=x/(1+x)
 return np.rint(np.clip(np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055),0,1)*255).astype(np.uint8)
def main(a):
 cv2.setNumThreads(2);a.output.mkdir(parents=True,exist_ok=False)
 c=json.loads((a.data/'complete.json').read_text());src=Path(c['source_master']);meta=json.loads((src/'transforms.json').read_text());frames={f['physical_camera']:f for f in meta['frames']};train={f['physical_camera']for f in meta['frames']if f['file_path']in meta['train_filenames']};rv=json.loads(Path(c['scalar_review']).read_text());trainEV={r['camera']:r['scalar_correction_EV']for r in rv['camera_rows']};assert set(trainEV)==train and c['eval_rgb_used_to_fit_gains']is False
 names=sorted(train);origins=np.array([frames[n]['transform_matrix']for n in names])[:,:3,3];EV=np.array([trainEV[n]for n in names]);rows=[]
 for r in c['rows']:
  n=r['camera'];f=frames[n];assert Path(r['source'])==src/f['file_path']
  old=cv2.imread(r['source'],cv2.IMREAD_UNCHANGED);newpath=a.data/'linear_rec709_exr'/r['linear_file'];new=cv2.imread(str(newpath),cv2.IMREAD_UNCHANGED);pngpath=a.data/'sdr_png'/r['png_file'];png=np.array(Image.open(pngpath));expected=old*(2**r['exposure_correction_ev']);linear_exact=np.array_equal(new,expected);png_exact=np.array_equal(png,encode(expected[...,::-1]));assert linear_exact and png_exact
  assert sha(r['source'])==r['source_sha256']and sha(newpath)==r['linear_sha256']and sha(pngpath)==r['png_sha256']
  if n in train:ev_expected=trainEV[n];method='TRAINtrackfit'
  else:
   pos=np.array(f['transform_matrix'])[:3,3];dist=np.linalg.norm(origins-pos,axis=1);ids=np.argsort(dist)[:c['interpolation_selected']['k']];w=np.maximum(dist[ids],1e-8)**(-c['interpolation_selected']['power']);w/=w.sum();ev_expected=float(w@EV[ids]);method='PoseOnlyNoTargetRGB';assert [names[i]for i in ids]==c['eval_gain_predictions'][n]['neighbors'];assert np.allclose(w,c['eval_gain_predictions'][n]['weights'],rtol=0,atol=1e-14)
  assert abs(ev_expected-r['exposure_correction_ev'])<1e-14
  rows.append(dict(camera=n,split='train'if n in train else'eval',EV=r['exposure_correction_ev'],linear_multiplier=2**r['exposure_correction_ev'],method=method,EXR_exact=linear_exact,PNG_exact=png_exact,source_and_output_hashes_verified=True))
  print(n,'exact',flush=True)
 # The numeric camera geometry/splits must be identical after filename-extension normalization.
 scalar=json.loads((a.data/'sdr_png/transforms.json').read_text());base=json.loads((src/'transforms.json').read_text())
 def normalize(m):
  for f in m['frames']:f['file_path']=str(Path(f['file_path']).with_suffix('.png'))
  for key in ['train_filenames','val_filenames','test_filenames']:m[key]=[str(Path(p).with_suffix('.png'))for p in m[key]]
  return m
 assert normalize(base)==scalar
 raw=Path('/fsx/oregon/gta_marshawn_2_1/cedar/tiff16/by_local');availability=[]
 for n in sorted(frames):
  for glob,loc in[(1700,448),(1740,488)]:
   p=raw/n/f'{n}.{loc:06d}.tif';availability.append(dict(camera=n,global_frame=glob,local_frame=loc,path=str(p),exists=p.exists(),bytes=p.stat().st_size if p.exists()else None))
 report=dict(passed=True,kind='ReadOnly_EXACT_scalar_chain_replay',script_sha256=sha(__file__),contract_sha256=sha(a.data/'complete.json'),source_script_sha256=c['script_sha256'],all66_actual_EXR_PNG_and_hashes_exact=True,geometry_and_split_exact_after_extension_normalization=True,
 train_eval_share_chain='gain4 scene-linear Rec709 EXR -> multiply2**cameraEV ONCE -> Reinhard -> sRGB8',inverse_tonemap_used_for_dataset_creation=False,
 inverse_tonemap_only='Separate legacy common-display comparison helper; not used by scalar dataset generation/native training/eval',rows=rows,
 limitations=['Traintrackfit is confounded by view-dependent reflectance, visibility and local texture; it is not acquisitionISO or whitebalance calibration.','EvalcameraEV is pose interpolation, not an independent measured response; train LOO error119milliEV is not target-camera calibration proof.','Scalar leaves chromatic/WB and spatial/vignette differences unchanged.','Distinct-frame calibration requires static diffuse multiview correspondences, connected camera support and independent validation; floor specular/actors excluded.'],
 independent_frames_available=availability,independent_source_bytes_total=sum(r['bytes']or 0 for r in availability),no_new_conversion=True,no_new_gains_fitted=True,no_NN_updates=True)
 (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=True,images=len(rows),independent_files_available=sum(r['exists']for r in availability))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('/fsx/tmp/marshawn_001660_hd/color_variants/gain4_scalar_trainfit_poseeval'));p.add_argument('--output',type=Path,required=True);main(p.parse_args())
