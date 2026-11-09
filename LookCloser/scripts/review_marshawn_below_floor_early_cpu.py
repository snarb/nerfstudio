"""Saved early production guards: source, matched control and two prior strengths."""
import argparse,hashlib,json
from pathlib import Path
from PIL import Image,ImageDraw

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def panel(images,labels,crop,limit):
    tiles=[x.crop(crop)for x in images]
    for x in tiles:x.thumbnail(limit)
    w=max(x.width for x in tiles);h=max(x.height for x in tiles);out=Image.new('RGB',(len(tiles)*w,h+32),'#171717');d=ImageDraw.Draw(out)
    for i,(im,label)in enumerate(zip(tiles,labels)):out.paste(im,(i*w,32));d.text((i*w+2,3),label,fill='white')
    return out
def stack(images,path):
    out=Image.new('RGB',(max(x.width for x in images),sum(x.height for x in images)),'#171717');y=0
    for x in images:out.paste(x,(0,y));y+=x.height
    out.save(path,quality=95)
def main(c,root,attempts):
    terminal=attempts is None
    arms=['production_control0','production_prior1e4','production_prior1e3']
    arm_steps={a:int(json.loads((root/'runs'/a/'complete.json').read_text())['final_step'])if terminal else 126316+attempts for a in arms}
    out=root/'early_CPU_reviews'/('terminal'if terminal else 'attempts_'+str(attempts));out.mkdir(parents=True,exist_ok=True)
    ready=json.loads((c/'data/ready.json').read_text());regions=json.loads((c/'native_semantic_grid_20000/corrected_v2/review/region_crops.json').read_text())
    source=c/'independent_occupancy/semantic_quarter_distortion_final_geometry/quarter_distortion_rawmin/renders'
    folders=[root/'runs'/a/'renders'/f'step-{arm_steps[a]:09d}'for a in arms]
    metrics={a:json.loads((p/'metrics.json').read_text())for a,p in zip(arms,folders)}
    body=[];details=[];inputs={}
    for cam in ['DC','HC','KA']:
        r=next(x for x in ready['records']if x['camera']==cam);paths=[c/'data'/r['image'],source/(cam+'.png')]+[p/(cam+'.png')for p in folders]
        images=[Image.open(p).convert('RGB')for p in paths];labels=[cam+' GT',cam+' source126316']+[cam+' '+name+' +'+str(arm_steps[a]-126316)for name,a in zip(['control0','prior1e-4','prior1e-3'],arms)]
        if sha(paths[0])!=r['rgb_sha256']:raise ValueError('Frozen GT RGB changed')
        for p in paths:inputs[str(p)]=sha(p)
        crops=list(r['crops'].values());w,h=images[0].size;box=[max(0,min(v[0]for v in crops)-48),max(0,min(v[1]for v in crops)-32),min(w,max(v[2]for v in crops)+48),min(h,max(v[3]for v in crops)+48)]
        body.append(panel(images,labels,box,(340,570)))
        for name,box in regions[cam].items():
            tile=panel(images,[x+' '+name for x in labels],box,(430,260));tile.save(out/(cam+'_'+name+'.png'));details.append(tile)
    stack(body,out/'all3_body.jpg');stack(details,out/'all9_details.jpg')
    selected={a:{k:m['metrics'][k]for k in ['psnr','ssim','lpips']}for a,m in metrics.items()}
    control=selected['production_control0'];delta={a:{k:v[k]-control[k]for k in control}for a,v in selected.items()}
    summary=dict(attempts=attempts,absolute_steps=arm_steps,same_absolute_step=len(set(arm_steps.values()))==1,metrics=selected,
        per_view={a:m['by_view']for a,m in metrics.items()},source_files_sha256=inputs,
        no_image_space_erasing=True,no_new_render=True,quality_promotion=False,
        repeatability_floor=dict(per_view_PSNR_dB=.013160131107397888,SSIM=.00007578730583190918,LPIPS=.00002274196594953537),
        interpretation='Compare candidate effects against measured native repeatability and all3+9 source guards. Existing mixed-boundary/bag masks are not optical alphaGT.')
    summary['delta_vs_terminal_control_unmatched_steps'if terminal else 'delta_vs_matched_control']=delta
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(dict(output=str(out),metrics=selected,delta=delta)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--root',type=Path,required=True);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--attempts',type=int);g.add_argument('--terminal',action='store_true');a=p.parse_args();main(a.campaign,a.root,a.attempts)
