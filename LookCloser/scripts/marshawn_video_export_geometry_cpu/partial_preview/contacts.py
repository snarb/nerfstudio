from pathlib import Path
import json,time,hashlib
import numpy as np
from PIL import Image,ImageDraw
O=Path(__file__).parent;C=O.parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
while not(O/'complete.json').exists():time.sleep(10)
request=read(O/'path.json');outputs={};positions=[]
for page in range(2):
 full=Image.new('RGB',(1920,3*536),(24,24,24));heads=Image.new('RGB',(1536,3*410),(24,24,24));feet=Image.new('RGB',(2304,3*410),(24,24,24))
 for row,i in enumerate(range(page*3,page*3+3)):
  r=request['records'][i];frame=r['frame_id'];tf=read(C/'frames'/frame/'data/transforms.json');ic=next(x for x in tf['frames']if x['physical_camera']=='IC');target=r['camera'];scale=np.array([target['fl_x']/ic['fl_x'],target['fl_y']/ic['fl_y']]);offset=np.array([target['cx'],target['cy']])-.5-(np.array([ic['cx'],ic['cy']])-.5)*scale
  with np.load(C/'frames'/frame/'data/semantics/IC.npz')as z:mask=z['black']|z['grey']
  yy,xx=np.where(mask);bottom=yy.max();band=yy>bottom-25;footxy=np.array([np.median(xx[band]),bottom-25])*scale+offset;centres=[np.rint(np.array(v)*scale+offset).astype(int)for v in [[1286,355],[1310,495]]]
  positions.append(dict(index=i,frame=frame,approximate_head_centres=[v.tolist()for v in centres],approximate_feet_centre=footxy.tolist(),locator='IC source pixels transformed by target K; approximate small-arc visual locator, not depth projection'))
  for col,kind in enumerate(['lookcloser','hybrid']):
   p=O/'images/frames'/f'{i:04d}'/(kind+'.png');im=Image.open(p).convert('RGB');assert im.size==(3840,2024);small=im.copy();small.thumbnail((960,506));full.paste(small,(col*960,row*536+28));ImageDraw.Draw(full).text((col*960+6,row*536+7),f'{frame} t={i/24:.5f}s {kind}; nativeR0/s4/g.75',fill='white')
   for actor,(x,y)in enumerate(centres):heads.paste(im.crop((x-192,y-192,x+192,y+192)),((col*2+actor)*384,row*410+25));ImageDraw.Draw(heads).text(((col*2+actor)*384+5,row*410+5),f'{frame} {kind} '+['black head','grey head'][actor],fill='white')
   x,y=np.rint(footxy).astype(int);feet.paste(im.crop((x-576,y-240,x+576,y+144)),(col*1152,row*410+25));ImageDraw.Draw(feet).text((col*1152+5,row*410+5),f'{frame} {kind} feet; native pixels',fill='white')
 for name,img in [('fullframe',full),('native_heads',heads),('native_feet',feet)]:
  p=O/(f'first6_{name}_page{page+1}'+('.jpg'if name=='fullframe'else'.png'));img.save(p,**({'quality':95}if name=='fullframe'else{}));outputs[str(p)]=sha(p)
(O/'contacts_complete.json').write_text(json.dumps(dict(files_sha256=outputs,positions=positions,patches_no_resize=True,fullframe_thumbnails_only=True,frames=6,fps=24,duration_seconds=.25),indent=2)+'\n')
