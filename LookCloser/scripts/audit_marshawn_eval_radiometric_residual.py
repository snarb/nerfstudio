"""Read-only material-region EV residuals after exact inverse sRGB/Reinhard."""
from pathlib import Path
import json,hashlib
import numpy as np,cv2
from PIL import Image,ImageDraw
R=Path('/fsx/tmp/marshawn_001660_hd');S=R/'native_png_random10k_floor_band_PRIOR_STAGED';O=S/'radiometry14k_DIAGNOSTIC'
ROIS={
'DC':{'cardboard_A':[690,80,710,98],'cardboard_B':[540,80,570,98],'paper_marker_A':[58,19,112,74],'paper_marker_B':[740,16,780,69],'white_shirt':[716,226,740,254],'olive_cloth':[998,300,1030,325],'dark_cloth':[910,175,945,213],'floor_gray_A':[320,410,410,450],'floor_gray_B':[1350,440,1420,490],'floor_yellow':[1110,720,1200,780],'floor_gray_front':[900,850,970,920]},
'HC':{'cardboard_A':[1000,180,1020,245],'cardboard_B':[1420,255,1460,275],'paper_marker_A':[1420,171,1465,220],'paper_marker_B':[965,274,995,312],'white_shirt':[1465,375,1480,400],'olive_cloth':[1020,448,1060,480],'dark_cloth':[1070,310,1100,345],'floor_gray_A':[50,725,160,805],'floor_gray_B':[1320,700,1390,760],'floor_yellow':[1700,700,1790,780]},
'KA':{'cardboard_A':[1630,5,1660,40],'cardboard_B':[1715,120,1740,145],'paper_marker_A':[1415,0,1470,48],'paper_marker_B':[1555,0,1590,40],'olive_cloth':[1010,227,1040,252],'dark_cloth':[1100,135,1130,165],'blue_cloth':[454,605,490,645],'floor_suspect':[150,700,290,820],'floor_gray_A':[1380,540,1500,610],'floor_yellow':[1100,800,1200,870],'floor_gray_B':[1250,260,1320,310]}}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inverse(x):
 t=np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4);return t/np.maximum(1-t,1e-15)
def main():
 O.mkdir(exist_ok=True);contract=R/'color_variants/gain4_scalar_trainfit_poseeval/sdr_png/color_contract.json';c=json.loads(contract.read_text());predicted={q['camera']:q['exposure_correction_ev']for q in c['rows']if q['camera']in ROIS};stats=[];source=[]
 for name,rois in ROIS.items():
  file=S/'gate14k_HD'/f'eval_{name}_buffers.npz';a=np.load(file);gt=a['gt'].astype(float);p=a['rgb'].astype(float);alpha=a['opacity'];original=R/'color_variants/gain4_scalar_trainfit_poseeval/sdr_png/images'/f'eval_{name}.png';u8=np.array(Image.open(original).convert('RGB'));assert np.array_equal(np.rint(gt*255).astype(np.uint8),u8);assert np.max(abs(gt-u8.astype(np.float32)/255))==0
  linearGT=inverse(gt);linearPred=inverse(np.clip(p,0,1-1e-10));Ygt=linearGT@np.array([.2126,.7152,.0722]);Yp=linearPred@np.array([.2126,.7152,.0722]);Y=gt@np.array([.2126,.7152,.0722]);smooth=cv2.GaussianBlur(Y,(3,3),.6);gy,gx=np.gradient(smooth);grad=np.hypot(gx,gy);valid=(alpha>=.99)&(gt.min(2)>.04)&(p.min(2)>.04)&(gt.max(2)<.97)&(p.max(2)<.97)&(Ygt>.005)&(Ygt<10)&(grad<.015)
  ev=np.log2(np.maximum(Yp,1e-15)/np.maximum(Ygt,1e-15));evrgb=np.log2(np.maximum(linearPred,1e-15)/np.maximum(linearGT,1e-15));sheet=Image.new('RGB',(700,len(rois)*145),(24,24,24));draw=ImageDraw.Draw(sheet);maskglobal=np.zeros(valid.shape,bool)
  for j,(label,bbox)in enumerate(rois.items()):
   x0,y0,x1,y1=bbox;v=valid[y0:y1,x0:x1].copy();
   if label.startswith('paper'):v&=(gt[y0:y1,x0:x1].min(2)>.30)&((gt[y0:y1,x0:x1].max(2)-gt[y0:y1,x0:x1].min(2))<.12)
   v[:2]=False;v[-2:]=False;v[:,:2]=False;v[:,-2:]=False;maskglobal[y0:y1,x0:x1]|=v;values=ev[y0:y1,x0:x1][v];ch=evrgb[y0:y1,x0:x1][v];stat=dict(camera=name,label=label,bbox=bbox,accepted_pixels=int(v.sum()),material_group='floor'if label.startswith('floor')else'background_cardboard'if label.startswith('cardboard')else'paper'if label.startswith('paper')else'cloth',pose_only_input_correction_EV=float(predicted[name]))
   if len(values):stat.update(median_log2Y_pred_over_GT=float(np.median(values)),median_channel_EV=np.median(ch,axis=0).tolist(),Y_EV_percentiles=np.percentile(values,[10,25,50,75,90]).tolist(),channel_EV_IQR=(np.percentile(ch,75,axis=0)-np.percentile(ch,25,axis=0)).tolist(),encoded_median_residual=np.median((p-gt)[y0:y1,x0:x1][v],0).tolist())
   stats.append(stat)
   for col,(im,caption)in enumerate([(gt[y0:y1,x0:x1],'GT'),(p[y0:y1,x0:x1],'actual14k'),(np.repeat(v[...,None],3,2).astype(float),'validmask')]):
    im=Image.fromarray(np.rint(np.clip(im,0,1)*255).astype(np.uint8));im.thumbnail((210,105));sheet.paste(im,(col*230,j*145+36));draw.text((col*230+4,j*145+3),f'{label} {caption}',fill='white')
   draw.text((4,j*145+19),f'bbox{bbox} n={len(values)} YEV='+str(round(float(np.median(values)),3)if len(values)else None),fill='yellow')
  sheet.save(O/(name+'_material_contacts.jpg'),quality=95);source.append(dict(camera=name,raw_buffer=str(file),raw_buffer_sha256=sha(file),source_GT_sha256=sha(original),source_GT_float32_exact=True,all_eligible_pixel_count=int(valid.sum()),all_eligible_Y_EV_quantiles=np.percentile(ev[valid],[10,25,50,75,90]).tolist()))
 report=dict(checkpoint_sha256='e92e525fe7451ef9cc2959ecca57a09e68925b20143714bf463c54a6ed998ae9',contract_sha256=sha(contract),script_sha256=sha(__file__),rows=stats,sources=source,definition='EV=log2(Y(inverseReinhard(inverse_sRGB(pred)))/Y(inverseReinhard(inverse_sRGB(GT)))); Rec709Y=.2126R+.7152G+.0722B. Inverted values retain SAMEbakedgain4+fixedpercam scalar; no gain estimated/applied.',filters='alpha>=.99; allchannelsGT/pred>.04 and<.97; inverseYgt>.005,<10; smoothedGTluma gradient<.015;2pixelROIerosion;paperwhiteGT minchannel>.30 and max-min<.12 (diagnostic segmentation ONLY)',gain_correction_not_applied=True,diagnostic_only=True,no_training_or_output_changes=True,manual_region_labels_pending_full_contact_review=True);(O/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print('done',len(stats),'regions','inputposeEV',predicted)
if __name__=='__main__':main()
