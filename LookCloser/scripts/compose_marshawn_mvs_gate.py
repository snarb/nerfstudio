"""Separate unchanged GT/parent14k/MVS16k/safe13k native output panels."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from PIL import Image,ImageDraw
R=Path('/fsx/tmp/marshawn_001660_hd')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def image(x):return Image.fromarray(np.rint(np.clip(x,0,1)*255).astype(np.uint8))
def main(a):
 import torch
 from torchmetrics.functional import structural_similarity_index_measure
 from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
 torch.set_num_threads(2);lpips=LearnedPerceptualImagePatchSimilarity(normalize=True).eval();rows=[];sources={};data={};small=[];S=a.stage
 for c in ['DC','HC','KA']:
  pp=R/f'native_png_random10k_floor_band_PRIOR_STAGED/gate14k_HD/eval_{c}_buffers.npz';qp=S/f'gate16k_HD/eval_{c}_buffers.npz';sp=R/f'native_png_uniform_LR0005_STAGED/best13000_HD_predictions/{c}.png';p=np.load(pp);q=np.load(qp);safe=np.array(Image.open(sp).convert('RGB')).astype(np.float32)/255
  assert np.array_equal(p['gt'],q['gt']);assert q['gt'].shape==safe.shape==(1012,1920,3)
  values=[q['gt'],p['rgb'],q['rgb'],safe];labels=['GT fixed scalarPNG','parent14k core0','MVS16k core0','safe13k core0 PNG8'];data[c]=(values,p,q);panel=Image.new('RGB',(4*480,253+24),'black');draw=ImageDraw.Draw(panel)
  for col,(v,t)in enumerate(zip(values,labels)):panel.paste(image(v).resize((480,253),Image.Resampling.LANCZOS),(480*col,24));draw.text((480*col+4,4),c+' '+t,fill='white')
  panel.save(S/f'gate16k_{c}_overview.jpg',quality=95);small.append(panel);sources[c]=dict(parent=str(pp),parent_sha256=sha(pp),phase=str(qp),phase_sha256=sha(qp),safe13=str(sp),safe13_sha256=sha(sp),GT_float_exact=True)
 regions={'HC_floor':('HC',[850,650,1450,980]),'KA_floor':('KA',[850,650,1450,980]),'KA_blackROI':('KA',[150,670,320,830]),'KA_prior_boundary_margin':('KA',[0,552,428,994]),'DC_prior_boundary_margin':('DC',[1757,782,1920,1012])}
 for region,(c,bbox)in regions.items():
  x0,y0,x1,y1=bbox;values,p,q=data[c];v=[x[y0:y1,x0:x1]for x in values];h,w=v[0].shape[:2];panel=Image.new('RGB',(w*4,h+24),'black');draw=ImageDraw.Draw(panel)
  for col,(x,t)in enumerate(zip(v,labels)):panel.paste(image(x),(col*w,24));draw.text((col*w+3,4),t,fill='white')
  path=S/f'gate16k_{region}.png';panel.save(path);gt=torch.tensor(v[0]).permute(2,0,1)[None]
  for label,pred in [('parent14k',v[1]),('MVS16k',v[2])]:
   tensor=torch.tensor(pred).permute(2,0,1)[None]
   with torch.inference_mode():ssim=float(structural_similarity_index_measure(tensor,gt,data_range=1.));distance=float(lpips(tensor,gt))
   lpips.reset();mse=float((tensor-gt).square().mean());rows.append(dict(region=region,camera=c,bbox=bbox,model=label,PSNR=-10*np.log10(max(mse,1e-12)),SSIM=ssim,LPIPS=distance,mean_RGB_error=(pred-v[0]).mean((0,1)).tolist(),mean_opacity=float((p if label=='parent14k'else q)['opacity'][y0:y1,x0:x1].mean())))
 sheet=Image.new('RGB',(1920,sum(x.height for x in small)),'black');y=0
 for im in small:sheet.paste(im,(0,y));y+=im.height
 sheet.save(S/'gate16k_all3_GT_parent_MVS_safe.jpg',quality=95)
 (S/'gate16k_contact_receipt.json').write_text(json.dumps(dict(sources=sources,regions=regions,columns=labels,GT_compositing=False,source_native_policy='actualcheckpointcore0,dt.003 forparent14/MVS16; safe13 extractednative ns-eval8bitforvisualonly',script_sha256=sha(__file__)),indent=2)+'\n');(S/'gate16k_crop_metrics.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);main(p.parse_args())
