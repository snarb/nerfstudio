"""Fixed-source visual contact pages from saved RGB only; no image editing model."""
import argparse,json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tile(image,size):
 x=image.copy();x.thumbnail(size);return x

def main(data,models,out):
 out.mkdir(parents=True,exist_ok=False);rows={r['camera']:r for r in json.loads((data/'ready.json').read_text())['records']};sources={k:Path(v)for k,v in [s.split('=',1)for s in models]};hashes={};n=len(sources)+1;body=Image.new('RGB',(1440,n*420),'#202020');draw=ImageDraw.Draw(body)
 for col,cam in enumerate(['DC','HC','KA']):
  paths={'GT':data/rows[cam]['image'],**{k:v/(cam+'.png')for k,v in sources.items()}};images={k:Image.open(p).convert('RGB')for k,p in paths.items()};hashes.update({str(p):sha(p)for p in paths.values()});c=list(rows[cam]['crops'].values());box=[min(x[0]for x in c)-20,min(x[1]for x in c)-20,max(x[2]for x in c)+20,max(x[3]for x in c)+20]
  for row,(name,im)in enumerate(images.items()):
   t=tile(im.crop(box),(470,390));body.paste(t,(col*480+(480-t.width)//2,row*420+25));draw.text((col*480+5,row*420+5),cam+' '+name,fill='white')
  detail=Image.new('RGB',(n*300,1040),'#202020');dd=ImageDraw.Draw(detail)
  for row,(actor,part)in enumerate([('black','head'),('grey','head'),('black','feet'),('grey','feet')]):
   x0,y0,x1,y1=rows[cam]['crops'][actor];box=[x0,y0,x1,min(y1,y0+170)]if part=='head'else[x0,max(y0,y1-170),x1,y1]
   for j,(name,im)in enumerate(images.items()):
    t=tile(im.crop(box),(290,225));detail.paste(t,(j*300+(300-t.width)//2,row*260+28));dd.text((j*300+4,row*260+5),name+' '+actor+' '+part,fill='white')
  detail.save(out/(cam+'_hair_feet.jpg'),quality=95)
 body.save(out/'body_overview.jpg',quality=95)
 contours=Image.new('RGB',(n*270,3*300),'#202020');d=ImageDraw.Draw(contours)
 for row,(cam,x,y,label)in enumerate([('KA',1028,331,'grey knee contour'),('HC',1052,576,'grey leg contour'),('DC',850,85,'head halo')]):
  paths={'GT':data/rows[cam]['image'],**{k:v/(cam+'.png')for k,v in sources.items()}}
  for col,(name,p)in enumerate(paths.items()):
   im=Image.open(p).convert('RGB').crop((x-40,y-40,x+41,y+41)).resize((243,243),Image.Resampling.NEAREST);contours.paste(im,(col*270+10,row*300+40));d.text((col*270+5,row*300+5),name,fill='white');d.text((col*270+5,row*300+20),cam+' '+label,fill='white')
 contours.save(out/'contour9_context.png');(out/'receipt.json').write_text(json.dumps(dict(source_hashes=hashes,manual_review_pending=True,no_labels_changed=True,no_GPU=True),indent=2)+'\n');print(out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--model',action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.data,a.model,a.output)
