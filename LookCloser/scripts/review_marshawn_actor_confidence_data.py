"""Inspect actual confidence rasters/source bytes; no training authorization."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(a):
 a.output.mkdir(parents=True,exist_ok=False);m=json.loads(a.manifest.read_text());tf=json.loads((Path(m['data_dir'])/'transforms.json').read_text());names={r['file_path']for r in tf['frames']};splits={k:all(f in names for f in tf[k])for k in ['train_filenames','val_filenames','test_filenames']if k in tf};assert all(splits.values()),'Split paths must match actual frame file_paths';tot={k:0. for k in ['hard','W32','W64']};N=0;rows=[];panels=[]
 for r in m['rows']:
  assert sha(r['rgb'])==r['rgb_sha256'];assert sha(r['archive'])==r['archive_sha256'];z=np.load(r['archive']);q=z['proposal'];v=z['source_ineligible'];assert q.dtype==v.dtype==np.bool_ and not(q&v).any();assert np.array_equal(q,z['W64']>0)
  for p in tot:
   w=z[p];assert w.dtype==np.float32 and np.isfinite(w).all()and w.min()>=0 and w.max()<=1 and not(w[v]!=0).any()and not(w[~q]!=0).any()
   if r['split']=='TRAIN':tot[p]+=w.sum(dtype=np.float64)
  if r['split']=='TRAIN':N+=q.sum()
  assert np.all(z['hard']<=z['W32']+1e-6)and np.all(z['W32']<=z['W64']+1e-6)
  image=np.asarray(Image.open(r['rgb']).convert('RGB'));yy,xx=np.where(q);x0=max(0,int(xx.min())-35);x1=min(image.shape[1],int(xx.max())+36);y0=max(0,int(yy.min())-35);y1=min(image.shape[0],int(yy.max())+36);base=image[y0:y1,x0:x1];panel=Image.new('RGB',(1400,510))
  for j,p in enumerate(['RGB','hard','W32','W64']):
   im=base.copy()
   if p!='RGB':w=z[p][y0:y1,x0:x1];im=np.clip(im*.5+w[...,None]*np.array([0,127,50]),0,255).astype(np.uint8);im[v[y0:y1,x0:x1]]=[255,0,0]
   tile=Image.fromarray(im);tile.thumbnail((350,480));panel.paste(tile,(j*350+(350-tile.width)//2,28));ImageDraw.Draw(panel).text((j*350+4,4),r['camera']+' '+p,fill='white')
  panel.save(a.output/(r['camera']+'.jpg'),quality=95);panels.append(panel);rows.append({'camera':r['camera'],'range':[float(z['hard'].min()),float(z['W64'].max())],'proposal_pixels':int(q.sum()),'source_ineligible':int(v.sum()),'native_hair_added':r['native_hair_added_pixels']})
 for k in range(0,len(panels),4):
  im=Image.new('RGB',(1400,510*len(panels[k:k+4])))
  for j,p in enumerate(panels[k:k+4]):im.paste(p,(0,j*510))
  im.save(a.output/('page_%02d.jpg'%(k//4)),quality=95)
 measured={p:float(tot[p]/N)for p in tot}
 for p in tot:assert abs(measured[p]-m['normalizer_policy_train'][p])<1e-12
 for r in m['native_eval']:
  assert sha(r['rgb'])==r['rgb_sha256']and sha(r['mask'])==r['mask_sha256'];im=np.asarray(Image.open(r['mask']));assert im.shape==(4320,8192)and np.all((im==0)|(im==255))
 report={'numerical_checks_pass':True,'manual_visual_review':'PENDING','manifest_sha256':sha(a.manifest),'rows':rows,'actual_train_proposal_count':int(N),'actual_global_normalizers':measured,'native_eval_count':len(m['native_eval']),'split_paths_match':splits,'GT_RGB_bytes_unchanged':True,'source_ineligible_overrides_every_policy':True,'no_training_authorization':True};(a.output/'numerical_QA.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['actual_global_normalizers']))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
