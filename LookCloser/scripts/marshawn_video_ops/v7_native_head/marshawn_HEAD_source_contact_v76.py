from pathlib import Path
import json,sys,concurrent.futures
import numpy as np
from PIL import Image,ImageDraw
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');sys.path.insert(0,str(V/'code/LookCloser/scripts'));from prepare_marshawn_tuning import sha,write
frame=int(sys.argv[1]);f=V/'frames'/f'{frame:06d}';O=V/'operations'/f'{frame}_head_ALL66_ROI_v73';out=V/'operations'/f'{frame}_current_HEAD_source_review_v76';out.mkdir(exist_ok=True);cameras=json.loads((V/'source_inventory.json').read_text())['cameras']
def tile(cam):
 source=json.loads((V/'sources'/str(frame)/cam/'receipt.json').read_text());spec=json.loads((O/cam/'spec.json').read_text());x0,y0,x1,y1=spec['crop_xyxy'];raw=np.array(Image.open(source['raw_native_image']).convert('RGB').crop((x0,y0,x1,y1)))
 p=f/'native_masks'/(cam+'.npz');meta=json.loads(p.with_suffix('.json').read_text());assert sha(p)==meta['archive_sha256']
 with np.load(p)as z:B=z['black'][y0:y1,x0:x1];G=z['grey'][y0:y1,x0:x1];H=z['qualified_actor_rgb'][y0:y1,x0:x1]if'qualified_actor_rgb'in z.files else np.zeros_like(B)
 with np.load(f/'data/admission_unknown_hair'/cam/'unknown_hair_native.npz')as z:U=z['unknown_hair_native'][y0:y1,x0:x1]
 images=[raw.copy(),raw.copy(),raw.copy()];images[1][B]=(images[1][B]*.45+np.array([0,210,0])*.55).astype('uint8');images[1][G]=(images[1][G]*.45+np.array([0,100,255])*.55).astype('uint8');images[1][H]=(images[1][H]*.4+np.array([0,255,255])*.6).astype('uint8');images[2][U]=(images[2][U]*.5+np.array([255,0,255])*.5).astype('uint8');result=Image.new('RGB',(1008,360));draw=ImageDraw.Draw(result)
 for j,im in enumerate(images):result.paste(Image.fromarray(im).resize((336,336)),(j*336,24));draw.text((j*336+3,3),cam+' '+['SOURCE','ACTORS + qualified RGB','admission UNKNOWN only'][j],fill='white')
 return cam,result,dict(camera=cam,native_sha256=sha(p),source_receipt_sha256=sha(V/'sources'/str(frame)/cam/'receipt.json'),hair_geometry_native_sha256=sha(f/'data/admission_unknown_hair'/cam/'unknown_hair_native.npz'),ROI=spec['crop_xyxy'])
with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:rows=list(pool.map(tile,cameras))
files={}
for page in range((len(rows)+11)//12):
 selected=rows[page*12:(page+1)*12];sheet=Image.new('RGB',(2016,2160))
 for i,(_,im,_)in enumerate(selected):sheet.paste(im,(i%2*1008,i//2*360))
 p=out/f'page_{page:02d}.jpg';sheet.save(p,quality=94);files[str(p)]=sha(p)
write(out/'review_receipt.json',dict(frame=frame,cameras=66,records=[x[2]for x in rows],visual_files_sha256=files,source_pixels_only_no_DATA_or_training_changes=True));print(out,flush=True)
