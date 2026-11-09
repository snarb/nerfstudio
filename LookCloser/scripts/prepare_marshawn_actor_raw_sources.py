from pathlib import Path
import json,sys,hashlib
import numpy as np,cv2,tifffile
from PIL import Image,ImageDraw
R=Path('/fsx/tmp/marshawn_001660_hd');O=R/'two_actor_isolation/raw_native_sources';O.mkdir(exist_ok=False);sys.path[:0]=[str(R/'code_native'),str(Path.cwd()/'scripts')]
from prepare_marshawn_color_variants import vendor,RAW,RIG,srgb,sha
from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_text,read_images_text
cams=read_cameras_text(RIG/'cameras.txt');ims=read_images_text(RIG/'images.txt');tf=json.loads((R/'color_variants/radial_k4_joint_scalar_DIAGNOSTIC/sdr_png/transforms.json').read_text());pq=vendor();geo=json.loads((R/'two_actor_isolation/geometry_precheck/review.json').read_text());by={f['physical_camera']:f for f in tf['frames']};gr={q['camera']:q for q in geo['records']};records=[]
for name in [f['physical_camera'] for f in tf['frames']]:
 f=by[name];c=cams[next(x.camera_id for x in ims.values()if x.name.startswith(name+'/'))];fx,fy,cx,cy,*dist=c.params;K=np.array([[fx,0,cx-.5],[0,fy,cy-.5],[0,0,1.]])
 uv=np.array(gr[name]['projected_boxes']['loose']);u=np.c_[ (uv[:,0]-f['cx'])/f['fl_x'],(uv[:,1]-f['cy'])/f['fl_y'],np.ones(8)];pr=cv2.projectPoints(u,np.zeros(3),np.zeros(3),K,np.array(dist))[0].reshape(-1,2);lo=np.maximum(np.floor(pr.min(0)-200).astype(int),0);hi=np.minimum(np.ceil(pr.max(0)+200).astype(int),[8192,4320]);src=RAW/name/(name+'.000408.tif');raw=tifffile.memmap(src,mode='r');clip=np.asarray(raw[lo[1]:hi[1],lo[0]:hi[0]]);lin=np.empty(clip.shape,np.float32);M=pq.rgb_to_rgb_matrix('rwg','rec709')
 for y in range(0,len(clip),128):lin[y:y+128]=np.maximum(np.maximum(pq.log3g10_decode(clip[y:y+128].astype(np.float64)/65535),0)@M.T,0).astype(np.float32)
 display=srgb(lin*4.);p=O/(name+'_RAW_native_crop.png');Image.fromarray(display).save(p)
 # Native source only, no resizedmaskcoordinate is reused. Model output maps back by integercropoffset.
 row={'camera':name,'split':'TRAIN'if f['file_path']in tf['train_filenames']else'EVAL','raw_source':str(src),'raw_sha256':sha(src),'native_source_raster':[8192,4320],'coordinate_system':'RAW_lens_distorted_arrayindices','crop_xyxy':[int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])],'crop_size':[int(hi[0]-lo[0]),int(hi[1]-lo[1])],'image':str(p),'image_sha256':sha(p),'display':'vendorLog3G10RWG→positiveRec709linear→globalgain4ONLY→fixedReinhard+sRGB; no newperviewfit; semanticdisplayonlyNOTnewNNdata','HD_K':[f['fl_x'],f['fl_y'],f['cx'],f['cy']],'raw_OpenCV_K':K.tolist(),'distortion':list(dist),'exact_downsample_ratios':[1920/8192,1012/4320]};records.append(row);print(name,row['crop_size'],flush=True)
(O/'request.json').write_text(json.dumps({'records':records,'raw_master_reusable_under_calibration_changes':True,'fullmask_target_raster':[8192,4320],'HD_target':[1920,1012],'method':'NativeRAWcroppilot; infernonlinearsegmentationinRAWspace, laterexactexistingprefilter/remap toHD. No naive resize ofdistortedrawmasktoUNDISTORTEDHD','preserved_prep_sha256':sha(R/'data/preparation.json'),'source_transforms_sha256':sha(R/'data/transforms.json'),'script_sha256':sha(str(Path(__file__))),'no_training':True},indent=2)+'\n')
