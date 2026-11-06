"""TRAIN-only conservative raised-mat envelopes. UNKNOWN, never actor positive/depth."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import cv2
from PIL import Image,ImageDraw
cv2.setNumThreads(2)
O=Path(__file__).resolve().parent;B=Path('/fsx/tmp/marshawn_001660_hd');T=B/'tuning_20261106/data';R=Path('/fsx/oregon/gta_marshawn_2_1_prep_20261028/root/Marshawn_2_1/golden/rig')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
# Manually reviewed in all63 full-frame TRAIN source thumbnails (480x253).
# Envelopes deliberately include adjacent ambiguous floor; no precise boundary claim.
boxes={
'AA':[208,150,340,253],'AB':[183,153,340,253],'AC':[185,193,278,253],'AD':[90,180,310,253],
'BA':[123,135,292,253],'BB':[105,139,261,253],'BC':[130,190,270,253],'BD':[130,215,260,253],
'CA':[73,154,292,253],'CB':[65,154,273,253],'CC':[98,210,251,253],'CD':[125,230,238,253],'CE':None,
'DA':[95,100,265,215],'DB':[85,136,280,244],'DD':[18,215,208,253],'DE':[0,218,225,253],
'EA':[63,108,226,207],'EB':[48,135,221,247],'EC':[0,183,170,253],'ED':[0,194,158,249],'EE':[0,228,157,253],
'FA':[70,86,209,172],'FB':[0,93,174,186],'FC':[0,170,188,248],'FD':[0,198,185,251],'FE':[0,223,154,253],
'GA':[72,97,202,163],'GB':[46,104,205,166],'GC':[0,170,191,218],'GD':[0,176,183,225],'GE':[0,207,155,244],
'HA':[70,88,201,144],'HB':[78,54,216,113],'HD':[52,155,208,202],'HE':[63,200,202,242],
'IA':[76,72,218,142],'IB':[85,88,222,155],'IC':[119,123,273,180],'ID':[78,144,226,193],'IE':[91,194,209,231],
'JA':[73,64,197,125],'JB':[106,89,244,155],'JC':[172,130,299,187],'JD':[110,166,263,212],'JE':None,
'KB':[109,74,228,137],'KC':[134,120,271,188],'KD':[161,169,288,212],
'LA':[144,51,233,120],'LB':[146,65,245,130],'LC':[210,136,302,179],
'MA':[228,63,301,122],'MB':[219,90,302,141],'MC':[262,123,336,162],'MD':[229,206,298,238],
'NA':[254,77,308,142],'NB':[267,75,314,133],'NC':[296,129,341,173],'ND':[278,161,335,200],'NE':None,
'OA':[270,90,340,154],'OD':[255,153,354,204]}
meta=json.loads((T/'transforms.json').read_text());frames=[f for f in meta['frames']if f['file_path']in meta['train_filenames']];assert set(boxes)=={f['physical_camera']for f in frames};assert not(set(boxes)&{'DC','HC','KA'})
man=json.loads((B/'two_actor_isolation/confidence_native_hair_v2/manifest.json').read_text());source={r['camera']:r for r in man['rows']}
cameras={}
for line in (R/'cameras.txt').read_text().splitlines():
 if not line or line.startswith('#'):continue
 a=line.split();assert a[1]=='OPENCV';cameras[int(a[0])]=np.array(a[4:],float)
lines=[l for l in (R/'images.txt').read_text().splitlines()if not l.startswith('#')];camids={}
for line in lines[::2]:
 a=line.split();camids[a[9].split('/')[0]]=int(a[8])
(O/'masks').mkdir(exist_ok=True);records=[];panels=[];started=time.time()
for f in frames:
 name=f['physical_camera'];bbox=boxes[name];fx,fy,cx,cy,k1,k2,p1,p2=cameras[camids[name]];K=np.array([[fx,0,cx-.5],[0,fy,cy-.5],[0,0,1.]])
 native=np.zeros((4320,8192),np.uint8);rawpoly=[];hdpoly=[]
 if bbox is not None:
  x0,y0,x1,y1=np.array(bbox,float)*4;hdpoly=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
  # Dense edges preserve nonlinearity under inverse RAW distortion.
  contour=np.concatenate([a+(b-a)*np.linspace(0,1,64)[:,None]for a,b in zip(hdpoly,np.roll(hdpoly,-1,axis=0))]);x=(contour[:,0]-f['cx'])/f['fl_x'];y=(contour[:,1]-f['cy'])/f['fl_y'];rr=x*x+y*y;rad=1+k1*rr+k2*rr*rr;xd=x*rad+2*p1*x*y+p2*(rr+2*x*x);yd=y*rad+p1*(rr+2*y*y)+2*p2*x*y;rawpoly=np.stack([fx*xd+cx-.5,fy*yd+cy-.5],1);cv2.fillPoly(native,[np.rint(rawpoly).astype(np.int32)],1)
  # Native edge uncertainty, once; 16RAWpx ~=4HDpx, distinct from semantic pad1.
  native=cv2.dilate(native,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(33,33)))
 dst=np.array([[f['fl_x'],0,f['cx']-.5],[0,f['fl_y'],f['cy']-.5],[0,0,1.]])
 mx,my=cv2.initUndistortRectifyMap(K,np.array([k1,k2,p1,p2]),None,dst,(1920,1012),cv2.CV_32FC1)
 small=cv2.resize(native.astype(np.float32),(1920,1012),interpolation=cv2.INTER_AREA);coverage=cv2.remap(small,(mx+.5)*(1920/8192)-.5,(my+.5)*(1012/4320)-.5,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);mask=coverage>0
 path=O/'masks'/(name+'.npz');np.savez_compressed(path,mat_unknown=mask)
 with np.load(source[name]['raw_archive'])as z:crop=z['crop_xyxy'].tolist()
 im=Image.open(T/f['file_path']).convert('RGB');arr=np.array(im);arr[mask]=(arr[mask]*.65+np.array([0,220,255])*.35).astype(np.uint8);im=Image.fromarray(arr).resize((480,253));panel=Image.new('RGB',(480,280));panel.paste(im,(0,24));ImageDraw.Draw(panel).text((3,5),name+' UNKNOWN mat envelope'+(' / no visible boundary'if bbox is None else ''),fill='white');panels.append(panel)
 rec=dict(camera=name,mask_path=str(path),mask_sha256=sha(path),mask_key='mat_unknown',shape=[1012,1920],dtype='bool',unknown_pixels=int(mask.sum()),source_HD=str(T/f['file_path']),source_HD_sha256=sha(T/f['file_path']),source_raw_crop=str(B/'two_actor_isolation/raw_native_sources'/(name+'_RAW_native_crop.png')),native_crop_xyxy=crop,source_raw_archive=source[name]['raw_archive'],source_raw_archive_sha256=source[name]['raw_archive_sha256'],authoring_bbox_480x253=bbox,HD_polygon=[]if bbox is None else hdpoly.tolist(),native_polygon_full_RAW=[]if bbox is None else rawpoly.tolist(),native_uncertainty_radius_pixels=16,raw_camera_id=camids[name],review='Full-frame TRAIN source viewed; conservative envelope includes neighboring floor; not a precise segmentation or measureddepth.',native_detail_reviewed=name in ['DE','EE'])
 if name in ['DE','EE']:
  rec['native_source_sha256']=sha(rec['source_raw_crop']);rawimage=Image.open(rec['source_raw_crop']).convert('RGB');a=np.array(rawimage);x0,y0,x1,y1=crop;rawmask=native[y0:y1,x0:x1]>0;a[rawmask]=(a[rawmask]*.65+np.array([0,220,255])*.35).astype(np.uint8);rawimage=Image.fromarray(a);rawimage.thumbnail((1000,1000));rawimage.save(O/(name+'_native_unknown_overlay.png'))
 records.append(rec)
for pg in range(4):
 sheet=Image.new('RGB',(1920,1120))
 for j,p in enumerate(panels[pg*16:pg*16+16]):sheet.paste(p,((j%4)*480,(j//4)*280))
 sheet.save(O/f'all_train_unknown_review_{pg}.jpg')
# Actual independently triangulated toe-cell contradiction footprints must be UNKNOWN.
for name,rect in [('DE',[736,936,754,949]),('EE',[556,957,574,970])]:
 with np.load(O/'masks'/(name+'.npz'))as z:assert z['mat_unknown'][rect[1]:rect[3],rect[0]:rect[2]].all()
manifest=dict(kind='TRAIN_ONLY_CONSERVATIVE_MAT_OCCLUDER_UNKNOWN_ENVELOPES',no_eval_masks_or_RGB_read=True,records=records,source_transforms_sha256=sha(T/'transforms.json'),raw_cameras_sha256=sha(R/'cameras.txt'),raw_images_sha256=sha(R/'images.txt'),builder_sha256=sha(__file__),coordinate_policy='Manually reviewed calibratedHD envelopes -> 256 dense contour points through explicit OPENCV inverse mapping -> full8192x4320RAW raster -> 16RAWpx uncertainty dilation once -> exact existingAREA-prefilter/single-remap -> any positive coverage UNKNOWN.',policy='Union into source_ineligible only. Never add to actor positives or remove existing support. No depth known: whole covered ray UNKNOWN, never certify empty before/behind mat.',known_limitations=['Envelope may include neighboring floor and actor pixels; conservative UNKNOWN may retain extra geometry.','Only DE/EE RAW detail reviewed; all63TRAIN HD overviews reviewed.','CE/JE/NE mat boundary not visible; no guessed mask. Existing other-person UNKNOWN still applies.','Does not certify all scene occluders, hidden mat boundaries, or physical visibility.'],measured_toe_footprint_regression_pass=True,total_unknown_pixels=sum(r['unknown_pixels']for r in records),seconds=time.time()-started)
(O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(records=len(records),total_unknown_pixels=manifest['total_unknown_pixels'],regression=True,seconds=manifest['seconds'])),flush=True)
