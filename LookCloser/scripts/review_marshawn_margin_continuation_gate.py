"""Fixed saved-image review pages; no labels, model queries or ranking changes."""
import argparse,json,hashlib,subprocess,sys
from pathlib import Path
from PIL import Image,ImageDraw

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(root,step):
 dest=root/'reviews'/f'step-{step:09d}';dest.mkdir(parents=True,exist_ok=False);data=root.parent/'data';rows={r['camera']:r for r in json.load(open(data/'ready.json'))['records']}
 arms={a:root/'review_archives'/a/f'step-{step:09d}'for a in ['original_mapped','source_margin002']}
 for path in arms.values():assert json.load(open(path/'fetch_verified.json'))['passed']
 metrics={a:json.load(open(p/'renders/metrics.json'))for a,p in arms.items()};regions={a:json.load(open(p/'region_diagnostics.json'))for a,p in arms.items()};bindings={}
 body=Image.new('RGB',(1440,1680),'#202020');bd=ImageDraw.Draw(body)
 for j,cam in enumerate(['DC','HC','KA']):
  sources={'GT':data/rows[cam]['image'],'parent126316':root/'zero_references/original_mapped/renders'/f'{cam}.png',**{a:p/'renders'/f'{cam}.png'for a,p in arms.items()}}
  images={a:Image.open(p).convert('RGB')for a,p in sources.items()};bindings.update({str(p):sha(p)for p in sources.values()});cs=list(rows[cam]['crops'].values());box=[min(c[0]for c in cs)-20,min(c[1]for c in cs)-20,max(c[2]for c in cs)+20,max(c[3]for c in cs)+20]
  for i,(arm,im)in enumerate(images.items()):
   tile=im.crop(box);tile.thumbnail((470,390));body.paste(tile,(j*480+(480-tile.width)//2,i*420+25));bd.text((j*480+7,i*420+5),cam+' | '+arm,fill='white')
  detail=Image.new('RGB',(1200,1040),'#202020');d=ImageDraw.Draw(detail)
  for row,(actor,part)in enumerate([('black','head'),('grey','head'),('black','feet'),('grey','feet')]):
   x0,y0,x1,y1=rows[cam]['crops'][actor];crop=[x0,y0,x1,min(y1,y0+170)]if part=='head'else[x0,max(y0,y1-170),x1,y1]
   for col,(arm,im)in enumerate(images.items()):
    tile=im.crop(crop);tile.thumbnail((290,225));detail.paste(tile,(col*300+(300-tile.width)//2,row*260+28));d.text((col*300+5,row*260+5),arm+' | '+actor+' '+part,fill='white')
  detail.save(dest/(cam+'_hair_feet.jpg'),quality=95)
 body.save(dest/'body_overview.jpg',quality=94)
 command=[sys.executable,str(Path(__file__).with_name('inspect_marshawn_exterior_opacity.py')),'--data',str(data),'--output',str(dest/'exterior')]
 for arm,p in arms.items():command+=['--arm',arm+'='+str(p/'renders')]
 subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
 summary=dict(step=step,attempts=step-126316,primary_metrics={a:{k:v for k,v in m['metrics'].items()if k!='validation_loss'}for a,m in metrics.items()},per_view={a:{view:{k:v for k,v in vals.items()if k!='validation_loss'}for view,vals in m['by_view'].items()}for a,m in metrics.items()},per_actor={a:{actor:{k:v for k,v in vals.items()if k!='validation_loss'}for actor,vals in m['by_actor'].items()}for a,m in metrics.items()},fixed_region_diagnostics={a:m['aggregate']for a,m in regions.items()},exterior16=[x for x in json.load(open(dest/'exterior/receipt.json'))['rows']if x['euclidean_margin_HD_pixels']==16],inputs_sha256=bindings,CPU_saved_images_only=True,manual_visual_review_pending=True,quality_promoted=False)
 (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(dest)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--step',type=int,required=True);a=p.parse_args();main(a.root,a.step)
