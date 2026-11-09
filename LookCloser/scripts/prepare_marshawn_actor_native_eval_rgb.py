"""Genuine native RAW-derived corrected eval RGB; never upscales HD targets."""
from pathlib import Path
import json,sys,os,hashlib
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import numpy as np,cv2,tifffile
from PIL import Image
R=Path('/fsx/tmp/marshawn_001660_hd')

def main():
 sys.path[:0]=[str(R/'code_native'),str(Path(__file__).parent)]
 from prepare_marshawn_color_variants import vendor,RAW,sha,srgb
 from marshawn_actor_coordinates import maps,target_intrinsics
 from marshawn_radial_photometry import radial_ev
 O=R/'two_actor_isolation/native_eval_RGB';O.mkdir(exist_ok=False);(O/'images').mkdir();data=R/'color_variants/radial_k4_joint_scalar_DIAGNOSTIC/sdr_png';tf=json.loads((data/'transforms.json').read_text());fitpath=R/'report/shared_radial1660_train_PRECHECK/fit_k4.json';fit=json.loads(fitpath.read_text());assert fit['uses_eval_RGB']is False and fit['k']==4;pq=vendor();M=pq.rgb_to_rgb_matrix('rwg','rec709');records=[];newframes=[];cv2.setNumThreads(2)
 for cam in ['DC','HC','KA']:
  f=next(x for x in tf['frames']if x['physical_camera']==cam);src=RAW/cam/(cam+'.000408.tif');raw=tifffile.memmap(src,mode='r');linear=np.empty(raw.shape,np.float32)
  for y in range(0,4320,128):linear[y:y+128]=np.maximum(np.maximum(pq.log3g10_decode(raw[y:y+128].astype(np.float64)/65535),0)@M.T,0).astype(np.float32)
  ev=float(fit['eval_predictions'][cam]['ev']);center=float(fit['radial_global_center_EV']);hdx,hdy=maps(f);small=cv2.resize(linear,(1920,1012),interpolation=cv2.INTER_AREA);hd=cv2.remap(small,(hdx+.5)*(1920/8192)-.5,(hdy+.5)*(1012/4320)-.5,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);hd=(hd.astype(np.float64)*np.exp2(radial_ev(f,4)-center+ev)[...,None]*4).astype(np.float32);hdPNG=srgb(hd);old=np.array(Image.open(data/f['file_path']).convert('RGB'));err=np.abs(hdPNG.astype(np.int16)-old.astype(np.int16));assert err.max()<=1
  full=dict(f,**target_intrinsics(f));mx,my=maps(full);native=cv2.remap(linear,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);del linear,mx,my,small,hd
  out=np.empty((4320,8192,3),np.uint8)
  for y0 in range(0,4320,128):
   y1=min(4320,y0+128);Y,X=np.ogrid[y0:y1,:8192];uv=np.stack(np.broadcast_arrays(X+.5,Y+.5),axis=-1);e=radial_ev(full,4,uv)-center+ev;native[y0:y1]=(native[y0:y1].astype(np.float64)*np.exp2(e)[...,None]*4).astype(np.float32);out[y0:y1]=srgb(native[y0:y1])
  p=O/'images'/('eval_'+cam+'.png');Image.fromarray(out).save(p);full['file_path']='images/'+p.name;full.pop('mask_path',None);newframes.append(full);records.append({'camera':cam,'RAW':str(src),'RAW_sha256':sha(src),'output':str(p),'output_sha256':sha(p),'native_raster':[8192,4320],'existing_HD_replay_max_code_error':int(err.max()),'existing_HD_replay_nonidentical_channels':int((err>0).sum()),'existing_HD_sourcePNG_sha256':sha(data/f['file_path']),'existing_pose_only_eval_EV':ev,'current_same_k4_global_center_EV':center,'native_linear_range':[float(native.min()),float(native.max())]});print(cam,records[-1],flush=True);del native,out
 (O/'cameras.json').write_text(json.dumps({'frames':newframes,'camera_model':'OPENCV','coordinate_system':'samepreparedworld','k1k2p1p2':0},indent=2)+'\n');(O/'receipt.json').write_text(json.dumps({'kind':'NATIVE_RAW_DERIVED_UNMODIFIED_ACTUAL_RGB_TARGETS','records':records,'source_transforms_sha256':sha(data/'transforms.json'),'existing_fit_sha256':sha(fitpath),'no_eval_fit':True,'global_gain4_applied_ONCE':True,'camera_EV_applied_ONCE':True,'radial_k4_applied_ONCE':True,'native_filter':'decodepositiveRec709→nativecalibbilinearremap→samegain4jointk4EV→Reinhard+sRGB','HD_filter':'verifiedexistingareaRAWlinearbeforemap;nativeandHDdifferentpixelfootprintsnotbitidenticaldownsamples','no_upscaling_of_HD':True,'semantic_mask_not_applied_to_GT':True,'optimizer_updates':0,'script_sha256':sha(__file__)},indent=2)+'\n')
if __name__=='__main__':main()
