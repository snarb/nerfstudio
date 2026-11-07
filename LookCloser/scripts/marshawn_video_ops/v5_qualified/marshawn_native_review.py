"""Read-only visual QA artifacts; never mutates prepared inputs or model state."""
from pathlib import Path
import argparse,json,time
import cv2,numpy as np
from PIL import Image,ImageDraw
from marshawn_actor_coordinates import raw_mask_coverage_to_HD
from prepare_marshawn_tuning import sha,write

def main(root,frame):
 cv2.setNumThreads(2);froot=root/'frames'/f'{frame:06d}';native=froot/'native_masks';out=froot/'native_review';out.mkdir(exist_ok=True)
 source=json.loads((root/'source_inventory.json').read_text());cameras=source['cameras'];tiles=[];stats=[]
 for camera in cameras:
  while not (native/(camera+'.json')).exists():time.sleep(5)
  meta=json.loads((native/(camera+'.json')).read_text());archive=native/(camera+'.npz')
  if sha(archive)!=meta['archive_sha256']:raise ValueError('Mask source changed '+camera)
  receipt=root/'sources'/str(frame)/camera/'receipt.json';row=json.loads(receipt.read_text());im=np.array(Image.open(row['image']).convert('RGB'))
  with np.load(archive) as z:masks={a:raw_mask_coverage_to_HD(z[a].astype('float32'),row['camera_row'])>=.5 for a in ['black','grey']};native_pixels=int(z['black'].sum()+z['grey'].sum())
  union=masks['black']|masks['grey'];yy,xx=np.where(union);tile=Image.new('RGB',(480,290),'#181818');d=ImageDraw.Draw(tile);d.text((5,4),f'{frame} {camera} B/G '+str([meta['identity'][a].get('visible_supported')for a in ['black','grey']]),fill='white')
  photo=Image.fromarray(im);photo.thumbnail((240,253));tile.paste(photo,(0,26))
  if len(xx):
   bbox=(max(0,int(xx.min())-24),max(0,int(yy.min())-24),min(im.shape[1],int(xx.max())+25),min(im.shape[0],int(yy.max())+25));ov=im.copy();ov[~union]//=5
   for a,color in [('black',[255,80,30]),('grey',[0,230,255])]:ov[masks[a]]=(ov[masks[a]]*.7+np.array(color)*.3).astype('uint8')
   crop=Image.fromarray(ov).crop(bbox);crop.thumbnail((240,253));tile.paste(crop,(240,26))
  tile.save(out/(camera+'_identity.jpg'));tiles.append(tile);stats.append(dict(camera=camera,native_actor_pixels=native_pixels,identity=meta['identity'],archive_sha256=sha(archive),source_receipt_sha256=sha(receipt),component_qualification=meta.get('component_qualification')))
  print(camera,flush=True)
 for start in range(0,len(tiles),12):
  page=Image.new('RGB',(1920,870),'#111111')
  for j,tile in enumerate(tiles[start:start+12]):page.paste(tile,(j%4*480,j//4*290))
  page.save(out/f'identity_page_{start//12+1:02d}.jpg',quality=95)
 largest=sorted(stats,key=lambda r:r['native_actor_pixels'],reverse=True)[:3]
 for rank,record in enumerate(largest):
  camera=record['camera'];row=json.loads((root/'sources'/str(frame)/camera/'receipt.json').read_text());raw=np.array(Image.open(row['raw_native_image']).convert('RGB'));z=np.load(native/(camera+'.npz'));sheet=Image.new('RGB',(2048,4224),'#111111');regions=[]
  for actor_index,actor in enumerate(['black','grey']):
   m=z[actor];y,x=np.where(m)
   if not len(x):continue
   for part_index,part in enumerate(['head','feet']):
    selection=y<np.percentile(y,18) if part=='head' else y>np.percentile(y,82);cx=int(np.median(x[selection]));cy=int(np.median(y[selection]));x0=min(max(cx-512,0),8192-1024);y0=min(max(cy-512,0),4320-1024);crop=raw[y0:y0+1024,x0:x0+1024].copy();local=m[y0:y0+1024,x0:x0+1024];ov=crop.copy();ov[local]=(ov[local]*.55+np.array([0,255,255])*.45).astype('uint8');offset=(actor_index*2+part_index)*1056;sheet.paste(Image.fromarray(crop),(0,offset+24));sheet.paste(Image.fromarray(ov),(1024,offset+24));ImageDraw.Draw(sheet).text((5,offset+4),f'{camera} {actor} {part} native1:1 [x{x0} y{y0}]',fill='white');regions.append(dict(actor=actor,part=part,bbox_native=[x0,y0,x0+1024,y0+1024]))
  sheet.save(out/f'native_rank{rank+1}_{camera}_head_feet.jpg',quality=96);record['native_review_regions']=regions
 write(out/'receipt.json',dict(frame=frame,cameras=len(stats),records=stats,largest_native_views=largest,scope='Source-bound identityandnativeboundaryvisualQA; nopixelorweightchanges; provisionaluntilrootreview',created=time.time()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--frame',type=int,required=True);a=p.parse_args();main(a.root,a.frame)
