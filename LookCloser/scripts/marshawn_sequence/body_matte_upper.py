"""Conservative Mat2 strict-only body boundary cleanup with explicit hair/prop guards."""
from pathlib import Path
import numpy as np,cv2,json
from PIL import Image,ImageDraw
import imageio.v2 as imageio
R=Path('/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109/operations');S=R/'flying_strict_low_20261112/neural';E=R/'body_matte_cleanup_20261112';O=E/'hybrid_body'
for d in ['factor','black','review']: (O/d).mkdir(parents=True,exist_ok=True)
# Coordinates in 480x270 review space. Each key: two standing-actor regions,
# bat-actor head, dread hair/head, bat prop guard, other tumbling actor guard.
K={
0: [[70,45,145,140],[210,70,305,170],[103,49,126,80],[220,76,267,119],[90,65,122,102],[218,45,310,120]],
5: [[70,45,140,145],[210,85,305,185],[92,47,118,77],[220,94,257,128],[88,66,125,99],[215,65,315,127]],
10:[[65,45,140,155],[220,100,310,185],[89,45,112,76],[220,119,255,155],[70,65,110,99],[215,45,335,150]],
15:[[65,50,145,165],[230,75,305,170],[88,52,112,83],[245,115,282,148],[80,87,122,122],[230,135,350,205]],
20:[[65,55,145,175],[225,60,310,165],[80,59,106,91],[242,62,285,110],[90,80,119,125],[230,145,335,205]],
25:[[70,65,150,185],[215,45,295,185],[88,68,110,94],[235,50,275,88],[73,90,145,124],[230,170,335,220]],
30:[[75,65,180,190],[190,35,285,175],[117,76,145,105],[213,34,256,81],[78,64,139,120],[225,158,340,225]],
35:[[85,70,180,200],[145,30,245,180],[117,75,146,105],[170,32,216,85],[103,92,176,133],[220,155,330,235]],
40:[[65,90,155,225],[125,30,210,215],[95,100,124,129],[141,32,196,91],[68,116,151,143],[230,165,360,250]],
45:[[45,80,190,265],[140,95,250,260],[124,82,153,113],[154,83,205,157],[0,165,86,244],[270,195,410,270]],
50:[[60,90,210,270],[185,115,310,270],[117,99,153,136],[191,116,244,164],[65,245,205,270],[340,250,440,270]],
55:[[50,105,250,270],[230,85,390,270],[102,109,153,153],[224,90,310,166],[150,257,251,270],[440,270,480,270]],
60:[[55,120,280,270],[235,80,425,270],[121,124,181,174],[250,85,347,176],[180,260,280,270],[440,270,480,270]],
65:[[60,110,320,270],[220,65,400,270],[155,122,221,181],[245,65,310,142],[220,260,320,270],[440,270,480,270]],
69:[[70,105,350,270],[230,40,390,270],[166,109,226,172],[252,43,310,121],[250,260,350,270],[440,270,480,270]],
}
yy,xx=np.mgrid[:1080,:1920];rows=[];v=imageio.get_writer(O/'comparison_24fps.mp4',fps=24,codec='libx264',quality=9,macro_block_size=2)
for i in range(70):
 lo=max(k for k in K if k<=i);hi=min(k for k in K if k>=i);t=0 if lo==hi else (i-lo)/(hi-lo);boxes=((1-t)*np.array(K[lo])+t*np.array(K[hi]))*4
 roi=np.zeros((1080,1920));guard=np.zeros_like(roi)
 for k,(x0,y0,x1,y1) in enumerate(boxes):
  signed=np.minimum.reduce([xx-x0,x1-xx,yy-y0,y1-yy])
  if k<2:roi=np.maximum(roi,np.clip(signed/12,0,1))
  else:guard=np.maximum(guard,np.clip((signed+20)/20,0,1))
 st=np.array(Image.open(S/f'{i:03}'/'strict.png'));old=np.load(S/f'{i:03}'/'strict.npz')['accumulation'][...,0].clip(0,1);ma=np.array(Image.open(E/f'sequence/alpha/{i:03}.png'))/255
 dist=cv2.distanceTransform((old>.5).astype('uint8'),cv2.DIST_L2,5)
 # Upper bodies only use at most 8px outer band; deeper12px lower-body hybrid remains.
 factor=1-roi*(1-guard)*np.clip((8-dist)/4,0,1)*(1-np.minimum(1,ma/np.maximum(old,1e-4)))
 lower=np.array(Image.open(E/f'hybrid/factor/{i:03}.png'))/65535
 factor=np.minimum(factor,lower)
 # Explicit guards override even lower-body result where their interpolation overlaps.
 factor=1-(1-factor)*(1-guard)
 factor=np.rint(factor*65535).astype('uint16');out=np.rint(st*(factor[...,None]/65535)).astype('uint8')
 Image.fromarray(factor).save(O/f'factor/{i:03}.png');Image.fromarray(out).save(O/f'black/{i:03}.png')
 protected=(guard==1)|(dist>=12);rows.append({'frame':i,'changed_pixels':int(np.any(out!=st,axis=2).sum()),'protected_max_rgb_change':int(np.abs(st.astype('int16')-out.astype('int16'))[protected].max()),'guard_boxes_xyxy':boxes[2:].tolist()})
 pan=Image.new('RGB',(1920,564));d=ImageDraw.Draw(pan)
 for k,(im,label) in enumerate([(st,'Previous strict'),(out,'Body matte / hair and bat guards')]):pan.paste(Image.fromarray(im).resize((960,540)),(960*k,24));d.text((960*k+3,4),f'{i:03} {label}',fill='white')
 v.append_data(np.array(pan))
 if i%5==0 or i==69:pan.save(O/f'review/{i:03}.jpg',quality=97)
 if i in [35,60]:
  box=(350,140,980,780) if i==35 else (930,520,1710,1080);w=box[2]-box[0];h=box[3]-box[1];p=Image.new('RGB',(w*2,h+24));p.paste(Image.fromarray(st).crop(box),(0,24));p.paste(Image.fromarray(out).crop(box),(w,24));ImageDraw.Draw(p).text((5,5),'Strict / conservative body',fill='white');p.save(O/f'review/native_{i:03}.jpg',quality=98)
v.close();(O/'receipt.json').write_text(json.dumps({'keyframe_coordinates_480x270':K,'upper_boundary_full_px':4,'upper_boundary_zero_px':8,'guard_feather_outside_px':20,'frames':rows},indent=2));print('COMPLETE',sum(x['changed_pixels'] for x in rows),max(x['protected_max_rgb_change'] for x in rows))
