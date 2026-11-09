"""All-camera actual-source foot review, independent of estimated mask hull.
Boot floor-guide locations only scope source crops; they are NOT observed
landmarks, source depth GT or automatically accepted ray witnesses.
"""
from pathlib import Path
import argparse,json,hashlib,itertools
import numpy as np
from PIL import Image,ImageDraw
R=Path('/fsx/tmp/marshawn_001660_hd');A=R/'two_actor_isolation'
GUIDES=np.array([[.050660123,.256559551,-.137509608],[-.129525821,.178813041,-.142318283],[.0249650794,.100941198,-.157207188],[-.000237964,.205308778,-.142720691]])
def proj(x,row):
 pose=np.asarray(row['transform_matrix']);v=(x-pose[:3,3])@pose[:3,:3];return np.c_[row['fl_x']*v[:,0]/-v[:,2]+row['cx']-.5,row['fl_y']*(-v[:,1])/-v[:,2]+row['cy']-.5]
def main(a):
 a.output.mkdir(parents=True,exist_ok=False);data=R/'color_variants/radial_k4_joint_scalar_DIAGNOSTIC/sdr_png';tf=json.loads((data/'transforms.json').read_text());boxes={}
 for n in [64,128,256]:
  p=a.experiment/('padding'+str(n))/'aabb.json'
  if p.exists():boxes[str(n)]=json.loads(p.read_text())['aabb']
 panels=[];records=[]
 for row in tf['frames']:
  cam=row['physical_camera'];path=data/row['file_path'];im=Image.open(path).convert('RGB');uv=proj(GUIDES,row);mn=np.maximum(np.floor(uv.min(0)-[60,100]),0).astype(int);mx=np.minimum(np.ceil(uv.max(0)+[60,45]),[im.width,im.height]).astype(int);
  if np.any(mx<=mn):mn=np.array([0,max(0,im.height-300)]);mx=np.array([im.width,im.height])
  crop=im.crop((*mn,*mx));d=ImageDraw.Draw(crop)
  for label,color in [('256','yellow'),('128','cyan'),('64','red')]:
   if label not in boxes:continue
   b=np.asarray(boxes[label]);corners=np.array(list(itertools.product(*zip(b[0],b[1]))));q=proj(corners,row)-mn
   for i in range(8):
    for j in range(i+1,8):
     if np.count_nonzero(corners[i]!=corners[j])==1:d.line([tuple(q[i]),tuple(q[j])],fill=color,width=2)
  for j,p in enumerate(uv):
   x,y=p-mn;d.ellipse((x-3,y-3,x+3,y+3),outline='lime',width=1);d.text((x+4,y),str(j),fill='lime')
  crop.thumbnail((500,340));panel=Image.new('RGB',(1120,370));panel.paste(crop,((500-crop.width)//2,25));ImageDraw.Draw(panel).text((4,4),cam+' SOURCEsole red64/cyan128/yellow256',fill='white')
  for j,(x,y)in enumerate(uv):
   patch=im.crop((int(x)-50,int(y)-50,int(x)+50,int(y)+50));patch=patch.resize((150,150));panel.paste(patch,(505+(j%2)*155,35+(j//2)*160));ImageDraw.Draw(panel).text((820+(j%2)*145,45+(j//2)*160),'guide'+str(j)+' NOTGT',fill='white')
  panel.save(a.output/(cam+'.jpg'),quality=95);panels.append(panel);records.append({'camera':cam,'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'guide_pixels_ARRAY_xy':uv.tolist(),'crop_xyxy':[*mn.tolist(),*mx.tolist()],'guide_in_source_raster':((uv[:,0]>=0)&(uv[:,0]<im.width)&(uv[:,1]>=0)&(uv[:,1]<im.height)).tolist(),'guide_visibility':'MANUAL_REVIEW_PENDING; out-of-FOV crop padding is not source background; do not count guides as measuredbootrays'})
 for k in range(0,len(panels),4):
  page=Image.new('RGB',(1120,370*len(panels[k:k+4])))
  for j,p in enumerate(panels[k:k+4]):page.paste(p,(0,j*370))
  page.save(a.output/('page_%02d.jpg'%(k//4)),quality=95)
 (a.output/'source_review_manifest.json').write_text(json.dumps({'rows':records,'mask_boxes':boxes,'guide_world_scope_only':GUIDES.tolist(),'guide_origin':'clusters of manuallyscoped AA/GA/OA boot approximatefloorhits; no boxused','all_images_unclipped_actual_source':True,'guide_NOT_depth_OR_observed_landmark_GT':True,'training_approval':False},indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
