"""Conservative lower-body MatAnyone2 cleanup; explicit protected upper bodies/props."""
from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw
import imageio.v2 as imageio
R=Path('/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109/operations'); S=R/'flying_strict_low_20261112/neural';E=R/'body_matte_cleanup_20261112';O=E/'hybrid'
for d in ['black','factor','review']: (O/d).mkdir(parents=True,exist_ok=True)
# Loose lower-body regions in native 1920x1080 output coordinates, not silhouette annotations.
keys={0:[[280,350,550,530],[850,430,1200,650]],10:[[330,400,510,580],[960,550,1200,750]],20:[[300,470,540,720],[920,450,1240,585]],30:[[430,490,730,750],[810,475,1140,630]],35:[[360,540,650,780],[675,450,860,700]],40:[[360,610,590,880],[530,550,815,860]],45:[[380,680,750,1040],[730,650,990,980]],50:[[300,990,800,1080],[800,900,1260,1080]],54:[[300,1080,800,1080],[800,1080,1260,1080]],69:[[300,1080,800,1080],[800,1080,1260,1080]]}
yy,xx=np.mgrid[:1080,:1920];rows=[];v=imageio.get_writer(O/'comparison_24fps.mp4',fps=24,codec='libx264',quality=9,macro_block_size=2)
for i in range(70):
 st=np.array(Image.open(S/f'{i:03}'/'strict.png'));old=np.load(S/f'{i:03}'/'strict.npz')['accumulation'][...,0].clip(0,1);ma=np.array(Image.open(E/'sequence/alpha'/f'{i:03}.png'))/255
 lo=max(k for k in keys if k<=i);hi=min(k for k in keys if k>=i);t=0 if lo==hi else (i-lo)/(hi-lo);boxes=(1-t)*np.array(keys[lo])+t*np.array(keys[hi]);roi=np.zeros(old.shape)
 for x0,y0,x1,y1 in boxes:
  roi=np.maximum(roi,np.clip(np.minimum.reduce([xx-x0,x1-xx,yy-y0,y1-yy])/16,0,1))
 dist=cv2.distanceTransform((old>.5).astype('uint8'),cv2.DIST_L2,5);boundary=np.clip((12-dist)/4,0,1)
 # Quantize factor once and use same factor for integration/reproduction.
 factor=1-roi*boundary*(1-np.minimum(1,ma/np.maximum(old,1e-4)))
 factor=np.rint(factor*65535).astype('uint16');out=np.rint(st*(factor[...,None]/65535)).astype('uint8')
 Image.fromarray(factor).save(O/'factor'/f'{i:03}.png');Image.fromarray(out).save(O/'black'/f'{i:03}.png')
 protected=(roi==0)|(dist>=12);rows.append({'frame':i,'changed_pixels':int(np.any(st!=out,axis=2).sum()),'protected_max_rgb_change':int(np.abs(st.astype('int16')-out.astype('int16'))[protected].max()),'boxes':boxes.tolist()})
 panel=Image.new('RGB',(1920,564));panel.paste(Image.fromarray(st).resize((960,540)),(0,24));panel.paste(Image.fromarray(out).resize((960,540)),(960,24));d=ImageDraw.Draw(panel);d.text((4,4),f'{i:03} Previous strict',fill='white');d.text((964,4),'Conservative lower-body cleanup; hair / bat protected',fill='white');v.append_data(np.array(panel))
 if i in [0,10,20,30,35,40,45,50,60,69]: panel.save(O/'review'/f'{i:03}.jpg',quality=95)
v.close();(O/'receipt.json').write_text(json.dumps({'region_keys':keys,'feather_pixels':16,'boundary_full_until_px':8,'boundary_zero_after_px':12,'frames':rows,'note':'Preserves source strict RGB outside lower-body regions and all >12px interior. No full semantic guarantee at ambiguous edges. Hair/prop/upper-body regions excluded. Last 16 frames unchanged.'},indent=2));print('COMPLETE',sum(r['changed_pixels'] for r in rows))
