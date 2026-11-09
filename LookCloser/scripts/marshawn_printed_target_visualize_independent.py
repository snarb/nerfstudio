#!/usr/bin/env python3
import json,cv2,numpy as np
from PIL import Image,ImageDraw
from marshawn_printed_target_independent import OUT,ROOT
old=json.loads((OUT/'freecorners_baseline_heldout_predictions.json').read_text());new=json.loads((OUT/'freecorners_refined_heldout_predictions.json').read_text());key=lambda z:(z['camera'],z['marker'],z['corner']);nd={key(z):z for z in new}
chosen=sorted(old,key=lambda z:z['error'],reverse=True)[:6]+[z for z in old if z['camera']=='HC'][:2]
sheet=Image.new('RGB',(4*400,2*380));draw=ImageDraw.Draw(sheet)
for j,z in enumerate(chosen):
 cam=z['camera'];a=cv2.imread(str(ROOT/cam/f'{cam}.000408.tif'),-1);g=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY);lo,hi=np.percentile(g[::8,::8],[1,99]);g=np.uint8(np.clip((g.astype(float)-lo)*255/(hi-lo),0,255));o=np.array(z['observed']);cx,cy=np.rint(o).astype(int);x0=max(0,cx-150);y0=max(0,cy-130);tile=Image.fromarray(g[y0:y0+300,x0:x0+400]).convert('RGB');d=ImageDraw.Draw(tile)
 for tag,v,color in [('observed',o,'lime'),('baseline',np.array(z['predicted']),'red'),('candidate',np.array(nd[key(z)]['predicted']),'cyan')]:
  x,y=v-[x0,y0];d.line((x-6,y,x+6,y),fill=color,width=1);d.line((x,y-6,x,y+6),fill=color,width=1)
 x=j%4*400;y=j//4*380;sheet.paste(tile,(x,y+55));draw.text((x+5,y+4),f'{cam} marker {z["marker"]} corner {z["corner"]}',fill='white');draw.text((x+5,y+20),f'baseline {z["error"]:.2f}px; candidate {nd[key(z)]["error"]:.2f}px',fill='white');draw.text((x+5,y+36),'green observed / red baseline / cyan candidate',fill='white')
sheet.save(OUT/'heldout_problem_patches.jpg')
