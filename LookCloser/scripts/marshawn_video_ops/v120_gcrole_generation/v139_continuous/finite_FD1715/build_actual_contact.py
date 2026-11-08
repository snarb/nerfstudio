from pathlib import Path
import json,sys
import numpy as np
from PIL import Image,ImageDraw
G=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108');sys.path.insert(0,str(G/'code/LookCloser/scripts'))
from prepare_marshawn_tuning import sha,write
O=G/'operations/1715_FD_components_current_positive';F=G/'frames/001715';done=json.loads((O/'application_complete.json').read_text());p=F/'native_masks/FD.npz';assert sha(p)==done['new_native_sha256']
s=json.loads((G/'sources/1715/FD/receipt.json').read_text());raw=np.array(Image.open(s['raw_native_image']).convert('RGB'))
with np.load(p)as z:new=z['black']|z['grey'];newH=z['qualified_actor_rgb']
with np.load(O/'before_application/0_FD.npz')as z:old=z['black']|z['grey']
D=new&~old;assert D.sum()==19881 and not(old&~new).any()
with np.load(O/'exact_CURRENT_RAW_PRE_photo283.npz')as z:P=z['unresolved_RAW_proposal']
y,x=np.where(new|D);box=(max(0,int(x.min())-80),max(0,int(y.min())-100),min(8192,int(x.max())+81),min(4320,int(y.max())+101));l,t,r,b=box
page=Image.new('RGB',(1800,1000),'#141414');draw=ImageDraw.Draw(page)
for i,(name,mask)in enumerate([('CURRENT SOURCE',None),('BEFORE actor',old),('EXACT current NN addition',D),('AFTER actor',new),('PHYSICAL UNKNOWN cardboard kept',P),('AFTER qualified RGB',newH)]):
 image=raw[t:b,l:r].copy()
 if mask is not None:
  m=mask[t:b,l:r];image[m]=(image[m]*.4+np.array([0,235,255])*.6).astype('uint8')
 im=Image.fromarray(image);im.thumbnail((600,455));ox=i%3*600;oy=i//3*500;page.paste(im,(ox,oy+30));draw.text((ox+5,oy+5),name,fill='white')
page.save(O/'FD_actual_before_delta_after.jpg',quality=95)
write(O/'actual_contact_proof.json',dict(frame=1715,camera='FD',native_sha256=sha(p),application_complete_sha256=sha(O/'application_complete.json'),source_native_image_sha256=s['raw_native_image_sha256'],crop_native_xyxy=box,actor_added19881=True,actor_removed0=True,contact=str(O/'FD_actual_before_delta_after.jpg'),contact_sha256=sha(O/'FD_actual_before_delta_after.jpg')))
print(O/'FD_actual_before_delta_after.jpg')
