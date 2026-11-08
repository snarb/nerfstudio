from pathlib import Path
import sys,json
import numpy as np
from PIL import Image,ImageDraw
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');G=Path('/fsx/tmp/marshawn_video_source_gcrole_g1_20261108');O=G/'operations/GC_corrected_native_visual';O.mkdir(exist_ok=True)
rows=[]
for f in[1695,1700,1705]:
 p=G/'frames'/f'{f:06d}'/'generation_corrections/GC_visible_CREW_hand/retracted_wrong_human_NN_component.npz'
 with np.load(p)as z:bad=z['wrong_CREW_actor_part']
 y,x=np.where(bad);x0=max(0,int(x.min())-180);x1=min(8192,int(x.max())+181);y0=max(0,int(y.min())-200);y1=min(4320,int(y.max())+201)
 src=json.loads((V/'sources'/str(f)/'GC/receipt.json').read_text());im=np.array(Image.open(src['raw_native_image']).convert('RGB').crop((x0,y0,x1,y1)))
 with np.load(V/'frames'/f'{f:06d}'/'native_masks/GC.npz')as z:before=(z['black']|z['grey'])[y0:y1,x0:x1]
 with np.load(G/'frames'/f'{f:06d}'/'native_masks/GC.npz')as z:after=(z['black']|z['grey'])[y0:y1,x0:x1]
 panels=[]
 for title,mask,color in[(f'{f} RAW',None,None),('old g0 actor',before,[255,0,255]),('retracted wrong CREW NN',bad[y0:y1,x0:x1],[255,100,0]),('new G1 actor',after,[0,240,255])]:
  a=im.copy()
  if mask is not None:a[mask]=(.45*a[mask]+.55*np.asarray(color)).astype('uint8')
  pic=Image.fromarray(a);pic.thumbnail((430,500));tile=Image.new('RGB',(430,540),'black');tile.paste(pic,(0,30));ImageDraw.Draw(tile).text((5,7),title,fill='white');panels.append(tile)
 row=Image.new('RGB',(1720,540));
 for i,pic in enumerate(panels):row.paste(pic,(i*430,0))
 row.save(O/f'GC{f}_native_before_delta_after.jpg');rows.append(row)
out=Image.new('RGB',(1720,540*3));
for i,row in enumerate(rows):out.paste(row,(0,i*540))
out.save(O/'ALL3_GC_native_correction.jpg');print(O/'ALL3_GC_native_correction.jpg')
