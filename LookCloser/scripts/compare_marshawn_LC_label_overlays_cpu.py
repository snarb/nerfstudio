"""Fixed post-render label-overlay comparison; no fitting or threshold selection."""
import argparse
import json
from pathlib import Path
from PIL import Image,ImageDraw
from prepare_marshawn_frozen_semantic_cpu import sha,write


def run(path,execute):
    spec=json.loads(path.read_text())
    if not execute:
        print(json.dumps(dict(CPU_only=True,CUDA_started=False,optimizer_created=False,phase='two_vs_four_UNKNOWN_physical_review')));return
    release=json.loads(Path(spec['root_release']).read_text())
    if release.get('reviewer')!='root' or release.get('LC_paired_prior_comparison_authorized')is not True or release['spec_sha256'].get(str(path))!=sha(path):
        raise ValueError('Exact root label-overlay comparison release required')
    for p,digest in release['bound_files'].items():
        if sha(p)!=digest:raise ValueError('Bound input changed: '+p)
    roots=[Path(spec['control']),Path(spec['candidate'])]
    render=[json.loads((p/'renders/complete.json').read_text())for p in roots]
    metrics=[json.loads((p/'visual_gate/metrics.json').read_text())for p in roots]
    if render[0]['source_pipeline_digest']!=render[1]['source_pipeline_digest'] or render[0]['selectors']!=render[1]['selectors'] or metrics[0]['ready_sha256']!=metrics[1]['ready_sha256']:
        raise ValueError('Fixed source/selector/GT contract differs')
    if [p['head_initial_probability']for p in render]!=[.01,.01]:raise ValueError('Initial prior must remain fixed')
    output=Path(spec['output']);output.mkdir(parents=True,exist_ok=False)
    ready_path=Path(spec['ready']);ready={r['camera']:r for r in json.loads(ready_path.read_text())['records']}
    regions=json.loads(Path(spec['regions']).read_text());body=[];details=[]
    for camera in ['DC','HC','KA']:
        source=ready_path.parent/ready[camera]['image']
        paths=[source,roots[0]/'renders'/camera/'hard_theta01.png',roots[1]/'renders'/camera/'hard_theta01.png',roots[0]/'renders'/camera/'hard_theta05.png',roots[1]/'renders'/camera/'hard_theta05.png']
        labels=['GT','original2UNK hard.1','qualified4UNK hard.1','original2UNK hard.5','qualified4UNK hard.5']
        images=[Image.open(p).convert('RGB')for p in paths]
        for region,crop in regions[camera].items():
            tiles=[image.crop(crop)for image in images];width,height=tiles[0].size
            panel=Image.new('RGB',(5*width,height+35),'#171717');draw=ImageDraw.Draw(panel)
            for i,(tile,label)in enumerate(zip(tiles,labels)):panel.paste(tile,(i*width,35));draw.text((i*width+2,3),camera+' '+label,fill='white')
            panel.save(output/(camera+'_'+region+'.png'));panel.thumbnail((2000,430));details.append(panel)
        crops=list(ready[camera]['crops'].values());crop=[max(0,min(v[0]for v in crops)-48),max(0,min(v[1]for v in crops)-32),min(1920,max(v[2]for v in crops)+48),min(1012,max(v[3]for v in crops)+48)]
        panel=Image.new('RGB',(5*330,560),'#171717');draw=ImageDraw.Draw(panel)
        for i,(image,label)in enumerate(zip(images,labels)):
            tile=image.crop(crop);tile.thumbnail((325,520));panel.paste(tile,(i*330,35));draw.text((i*330+2,3),camera+' '+label,fill='white')
        body.append(panel)
    for filename,panels in [('all_actors_label_overlay_pair.jpg',body),('all_details_label_overlay_pair.jpg',details)]:
        combined=Image.new('RGB',(max(p.width for p in panels),sum(p.height for p in panels)),'#171717');y=0
        for panel in panels:combined.paste(panel,(0,y));y+=panel.height
        combined.save(output/filename,quality=95)
    write(output/'comparison.json',dict(CPU_only=True,CUDA_started=False,optimizer_created=False,
        metric_comparison={name:{v:m['metrics']for v,m in doc['variants'].items()}for name,doc in zip(['original_2UNK','fit_4UNK'],metrics)},
        source_pipeline_same=True,initial_prior_same=True,source_GT_masks_and_selectors_same=True,
        render_completion_sha256={str(p/'renders/complete.json'):sha(p/'renders/complete.json')for p in roots},
        qualified_override_sha256=spec['qualified_override_sha256'],automatic_promotion=False,
        interpretation='Only two additional TRAIN positive confidences became UNKNOWN in the new fit. Same source/camera history/priors/physics. Existing verified original2UNK control reused. No new EVALlabel/threshold selection; native1024 physicalcoverage remains capped.'))
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();run(args.spec,args.execute)
