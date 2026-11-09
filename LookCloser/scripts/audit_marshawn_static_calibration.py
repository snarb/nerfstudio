"""Independentframe1700 static-background calibration diagnostic, no supplied-rig match filter."""
from pathlib import Path
import argparse,json,hashlib,itertools,sys
import numpy as np,cv2
from PIL import Image,ImageDraw
from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_text,read_images_text
from audit_marshawn_geometry import projection,fundamental,sampson
R=Path('/fsx/tmp/marshawn_001660_hd');O=R/'static_calibration1700';RAW=Path('/fsx/oregon/gta_marshawn_2_1/cedar/tiff16/by_local');RIG=Path('/fsx/oregon/gta_marshawn_2_1_prep_20261028/root/Marshawn_2_1/golden/rig');GROUPS={'DC':['DB','CC','EB','DD'],'HC':['HD','IB','ID','HB'],'KA':['KB','LB','LA','JA']};NAMES=list(dict.fromkeys(sum(([k]+v for k,v in GROUPS.items()),[])))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for q in iter(lambda:f.read(8<<20),b''):h.update(q)
 return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def prepare():
 O.mkdir(exist_ok=False);cv2.setNumThreads(2);cams=read_cameras_text(RIG/'cameras.txt');ims=read_images_text(RIG/'images.txt');rig={q.name.split('/')[0]:cams[q.camera_id]for q in ims.values()};tf=json.loads((R/'data/transforms.json').read_text());rows={q['physical_camera']:q for q in tf['frames']};records=[]
 for name in NAMES:
  src=RAW/name/f'{name}.000448.tif';raw=cv2.imread(str(src),cv2.IMREAD_UNCHANGED);assert raw.shape==(4320,8192,3);gray=np.empty(raw.shape[:2],np.uint8)
  for y in range(0,4320,128):
   x=raw[y:y+128].astype(np.float32)/65535;x=np.expm1(x*(np.log(10)/.224282))/155.975327-.01;x=np.maximum(x.mean(2),0)*4;x=x/(1+x);x=np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055);gray[y:y+128]=np.rint(np.clip(x,0,1)*255).astype(np.uint8)
  del raw
  fx,fy,cx,cy,*d=rig[name].params;K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]]);srcK=K.copy();srcK[:2,2]-=.5;dstK=K.copy();dstK[:2]*=.5;dstK[:2,2]-=.5;mx,my=cv2.initUndistortRectifyMap(srcK,np.array(d),None,dstK,(4096,2160),cv2.CV_32FC1);undist=cv2.remap(gray,mx,my,cv2.INTER_LINEAR);del gray,mx,my
  Image.fromarray(undist).save(O/(name+'_4k.png'));preview=Image.fromarray(undist).resize((1024,540),Image.Resampling.BOX);ImageDraw.Draw(preview).text((3,3),name+' global1700/local448 independent static selection',fill=255);preview.save(O/(name+'_preview.jpg'),quality=94);row=dict(rows[name]);row.update(w=4096,h=2160,fl_x=fx*.5,fl_y=fy*.5,cx=cx*.5,cy=cy*.5);records.append(dict(camera=name,row=row,source=str(src),source_sha256=sha(src),image_sha256=sha(O/(name+'_4k.png'))));save(O/'progress.json',dict(prepared=name));print('prepared',name,flush=True)
 save(O/'sources.json',dict(frame=1700,localframe=448,groups=GROUPS,records=records,transforms_sha256=sha(R/'data/transforms.json'),rig_sha256={q:sha(RIG/q)for q in ['images.txt','cameras.txt']},script_sha256=sha(__file__),no_target1660_RGB_used=True,source_pixel_relation='4KundistortedcanonicalK=.5rawK;OpenCVsource/destprincipalpoints−.5;SIFTtrueUV=keypoint+.5;errorsreported×2rawpixels'))
if __name__=='__main__':prepare()
