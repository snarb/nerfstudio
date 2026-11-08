from pathlib import Path
import os,json,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch,lpips
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
O=Path(__file__).parent;A=O.parents[1]/'operations/encoder_color_audit_v1';torch.set_num_threads(2);net=lpips.LPIPS(net='alex',verbose=False).cpu().eval();crop=[2000,700,2500,1300];records=[]
for i in range(3):
 reference=np.asarray(Image.open(A/'PNG_masters'/f'{i:04d}.png').convert('RGB').crop(crop),np.float32)/255
 for kind in ['baseline_default','explicit709_default','baseline_interpreted_HD709']:
  decoded=np.asarray(Image.open(A/kind/f'{i:04d}.png').convert('RGB').crop(crop),np.float32)/255
  with torch.inference_mode():value=net(torch.from_numpy(reference).permute(2,0,1)[None]*2-1,torch.from_numpy(decoded).permute(2,0,1)[None]*2-1).item()
  records.append(dict(frame=i,decoded=kind,PSNR=float(peak_signal_noise_ratio(reference,decoded,data_range=1)),SSIM=float(structural_similarity(reference,decoded,data_range=1,channel_axis=2)),LPIPS=value))
r=dict(kind='SAME_TWO_ENCODES_FIXED_PNG_REFERENCE_PATCH_METRICS_V1',crop_xyxy=crop,records=records,reference='OriginalPNGmasters; codecerroronly, notmodelqualityorheldoutGT.',CPU_only=True,additional_encodes=0)
(O/'codec_patch_metrics.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
