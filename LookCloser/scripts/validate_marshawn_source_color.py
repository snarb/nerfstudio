#!/usr/bin/env python3
"""Independent TIFF channel/decode/mapping check; never edits prepared images."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
import cv2,tifffile
from PIL import Image,ImageDraw

ROOT=Path('/fsx/tmp/marshawn_001660_hd');OUT=ROOT/'source_color_validation_HC1660';SRC=Path('/fsx/oregon/gta_marshawn_2_1/cedar/tiff16/by_local/HC/HC.000408.tif');PQ=Path('/fsx/oregon/gta_marshawn_2_1/six_sec_20261019/code/engine_gta_pv_export/ca4d/ca4d/pq.py');TWIN=Path('/fsx/oregon/gta_marshawn_2_1_prep_20261028/code/af_twin8.py')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
 cv2.setNumThreads(2);OUT.mkdir(exist_ok=False)
 with tifffile.TiffFile(SRC) as f:
  page=f.pages[0];raw=page.asarray();tags={k:str(page.tags[k].value) for k in ['ImageWidth','ImageLength','BitsPerSample','PhotometricInterpretation','Compression','SamplesPerPixel','PlanarConfiguration','Orientation','SampleFormat'] if k in page.tags};header=dict(shape=list(raw.shape),dtype=str(raw.dtype),photometric=str(page.photometric),samplesperpixel=page.samplesperpixel,bitspersample=page.bitspersample,planarconfig=int(page.planarconfig),tags=tags)
 cvraw=cv2.imread(str(SRC),cv2.IMREAD_UNCHANGED)[...,::-1]
 equality=np.array_equal(raw,cvraw);array_sha=hashlib.sha256(raw.tobytes()).hexdigest();cv_sha=hashlib.sha256(cvraw.tobytes()).hexdigest()
 print(json.dumps(dict(stage='raw_channels',tifffileRGB_equals_OpenCV_BGR_reversed=equality,array_sha256=array_sha,cv_array_sha256=cv_sha,header=header)),flush=True)
 assert equality
 pq=module('independent_ca4d_pq',PQ);ours=module('prepared_marshawn',Path('scripts/prepare_marshawn_hd.py'));matrix=pq.rgb_to_rgb_matrix('rwg','rec709');reinhard=np.empty(raw.shape,np.uint8);plain=np.empty(raw.shape,np.uint8)
 for y in range(0,len(raw),128):
  linear=np.maximum(pq.log3g10_decode(raw[y:y+128].astype(np.float64)/65535),0)@matrix.T;linear=np.maximum(linear,0)
  for dst,display in [(reinhard,linear/(1+linear)),(plain,np.minimum(linear,1.))]:
   srgb=np.where(display<=.0031308,12.92*display,1.055*np.power(display,1/2.4)-.055);dst[y:y+len(display)]=np.rint(np.clip(srgb,0,1)*255).astype(np.uint8)
 meta=json.loads((ROOT/'data/transforms.json').read_text());row=next(r for r in meta['frames'] if r['physical_camera']=='HC');h,w=row['h'],row['w'];sx=w/8192;sy=h/4320
 from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_text,read_images_text
 rig=Path('/fsx/oregon/gta_marshawn_2_1_prep_20261028/root/Marshawn_2_1/golden/rig');images=read_images_text(rig/'images.txt');im=next(i for i in images.values() if i.name.startswith('HC/'));cam=read_cameras_text(rig/'cameras.txt')[im.camera_id];fx,fy,cx,cy,k1,k2,p1,p2=cam.params
 # Independent explicit OPENCV distortion; no initUndistortRectifyMap used.
 yy,xx=np.mgrid[0:h,0:w];x=(xx+.5-row['cx'])/row['fl_x'];y=(yy+.5-row['cy'])/row['fl_y'];r2=x*x+y*y;rad=1+k1*r2+k2*r2*r2;xd=x*rad+2*p1*x*y+p2*(r2+2*x*x);yd=y*rad+p1*(r2+2*y*y)+2*p2*x*y
 native_u=fx*xd+cx-.5;native_v=fy*yd+cy-.5;mx=((native_u+.5)*sx-.5).astype(np.float32);my=((native_v+.5)*sy-.5).astype(np.float32)
 def remap(rgb):return cv2.remap(cv2.resize(rgb,(w,h),interpolation=cv2.INTER_AREA),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
 ref=remap(reinhard);plain_hd=remap(plain);prepared=np.array(Image.open(ROOT/'data'/row['file_path']).convert('RGB'));difference=ref.astype(int)-prepared.astype(int)
 raw_sample=raw[::37,::41];ours_linear=np.maximum(np.expm1((raw_sample.astype(np.float32)/65535)*(np.log(10)/.224282))/155.975327-.01,0);vendor_linear=np.maximum(pq.log3g10_decode(raw_sample.astype(float)/65535),0)
 sample_positions=[(320,800),(500,960),(750,1100),(100,100),(900,1600)];samples=[]
 for y,x in sample_positions:samples.append(dict(hd_xy=[x,y],mapped_raw_uv=[float(native_u[y,x]),float(native_v[y,x])],raw_nearest_RGB=raw[int(round(native_v[y,x])),int(round(native_u[y,x]))].tolist(),prepared_RGB=prepared[y,x].tolist(),independent_Reinhard_RGB=ref[y,x].tolist(),vendor_plain_RGB=plain_hd[y,x].tolist()))
 for name,rgb in [('prepared',prepared),('independent_vendor_decode_Reinhard',ref),('vendor_decode_plain_clip_sRGB',plain_hd),('difference_x16',np.clip(abs(difference)*16,0,255).astype(np.uint8))]:Image.fromarray(rgb).save(OUT/(name+'.png'))
 vendor_jpeg=Image.open('/fsx/oregon/gta_marshawn_2_1/cedar/prep/subpix/Marshawn_2_1/working/fullres/001660/images/HC_001660.jpg').convert('RGB')
 panels=[('Prepared: declared Reinhard',Image.fromarray(prepared)),('Independent vendor decode +Reinhard',Image.fromarray(ref)),('Vendor decode plain clip +sRGB',Image.fromarray(plain_hd)),('Existing vendor JPEG, different canvas/look',vendor_jpeg)]
 canvas=Image.new('RGB',(960,560));d=ImageDraw.Draw(canvas)
 for i,(label,image) in enumerate(panels):image.thumbnail((480,253));xx=i%2*480;yy=i//2*280;canvas.paste(image,(xx,yy+25));d.text((xx+3,yy+3),label,fill='white')
 canvas.save(OUT/'display_comparison.jpg',quality=92)
 report=dict(source=str(SRC),source_sha256=sha(SRC),header=header,tifffile_RGB_exact_equals_cv2_BGR_reversed=equality,rgb_array_sha256=array_sha,cv2_reversed_array_sha256=cv_sha,vendor_pq_source=str(PQ),vendor_pq_sha256=sha(PQ),vendor_twin_source=str(TWIN),vendor_twin_sha256=sha(TWIN),rwg_matrix_max_abs_difference=float(abs(matrix-ours.M).max()),sampled_linear_decode_max_abs_difference=float(abs(ours_linear-vendor_linear).max()),sampled_linear_decode_mean_abs_difference=float(abs(ours_linear-vendor_linear).mean()),independent_HD_reference=dict(max_abs_code_error=int(abs(difference).max()),mean_abs_code_error=float(abs(difference).mean()),fraction_identical_channels=float((difference==0).mean()),fraction_channel_error_over1=float((abs(difference)>1).mean())),plain_vs_Reinhard_mean_abs_code_difference=float(abs(plain_hd.astype(float)-prepared).mean()),samples=samples,source_encoding='REDWideGamut Log3G10 uint16 RGB; neither linear norPQ',display_difference='Our declared transform additionally applies Reinhard; vendor af_twin8 uses linear clip then sRGB, so ours is darker/compressed in highlights by design',geometry_reference='Explicit calibrated OPENCV distortion polynomial and half-pixel lattice; downsample-before-remap matches declared preparation contract',limitations=['Existing vendor JPEG has different undistorted canvas and possibly independent look; compared visually only','This verifies ingestion and display transform, not training/marching correctness'])
 (OUT/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
