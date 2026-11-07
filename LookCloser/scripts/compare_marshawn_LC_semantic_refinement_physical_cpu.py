"""Fixed base/coarse-refine/residual-refine physical comparison; no fitting."""
import argparse
import json
from pathlib import Path
from PIL import Image,ImageDraw
from prepare_marshawn_frozen_semantic_cpu import sha,write


def run(path,execute):
    spec=json.loads(path.read_text())
    if not execute:
        print(json.dumps(dict(CPU_only=True,CUDA_started=False,optimizer_created=False,phase='matched_refinement_physical_review')));return
    release=json.loads(Path(spec['root_release']).read_text())
    if release.get('reviewer')!='root' or release.get('LC_refinement_comparison_authorized')is not True or release['spec_sha256'].get(str(path))!=sha(path):
        raise ValueError('Exact root label-overlay comparison release required')
    for p,digest in release['bound_files'].items():
        if sha(p)!=digest:raise ValueError('Bound input changed: '+p)
    roots=[Path(spec[k])for k in ['base_control','coarse_refinement','residual_refinement']]
    render=[json.loads((p/'renders/complete.json').read_text())for p in roots]
    metrics=[json.loads((p/'visual_gate/metrics.json').read_text())for p in roots]
    if len({r['source_pipeline_digest']for r in render})!=1 or any(r['selectors']!=render[0]['selectors']for r in render) or len({r['ready_sha256']for r in metrics})!=1:
        raise ValueError('Fixed source/selector/GT contract differs')
    if [p['head_initial_probability']for p in render]!=[.01,.01,.01]:raise ValueError('Initial prior must remain fixed')
    output=Path(spec['output']);output.mkdir(parents=True,exist_ok=False)
    ready_path=Path(spec['ready']);ready={r['camera']:r for r in json.loads(ready_path.read_text())['records']}
    regions=json.loads(Path(spec['regions']).read_text());body=[];details=[]
    for camera in ['DC','HC','KA']:
        source=ready_path.parent/ready[camera]['image']
        paths=[source]+[root/'renders'/camera/(selector+'.png')for selector in ['hard_theta01','hard_theta05']for root in roots]
        labels=['GT']+[label+' '+threshold for threshold in ['hard.1','hard.5']for label in ['base7937','coarse refine','residual refine']]
        images=[Image.open(p).convert('RGB')for p in paths]
        for region,crop in regions[camera].items():
            tiles=[image.crop(crop)for image in images];width,height=tiles[0].size
            panel=Image.new('RGB',(7*width,height+35),'#171717');draw=ImageDraw.Draw(panel)
            for i,(tile,label)in enumerate(zip(tiles,labels)):panel.paste(tile,(i*width,35));draw.text((i*width+2,3),camera+' '+label,fill='white')
            panel.save(output/(camera+'_'+region+'.png'));panel.thumbnail((2000,430));details.append(panel)
        crops=list(ready[camera]['crops'].values());crop=[max(0,min(v[0]for v in crops)-48),max(0,min(v[1]for v in crops)-32),min(1920,max(v[2]for v in crops)+48),min(1012,max(v[3]for v in crops)+48)]
        panel=Image.new('RGB',(7*300,560),'#171717');draw=ImageDraw.Draw(panel)
        for i,(image,label)in enumerate(zip(images,labels)):
            tile=image.crop(crop);tile.thumbnail((295,520));panel.paste(tile,(i*300,35));draw.text((i*300+2,3),camera+' '+label,fill='white')
        body.append(panel)
    for filename,panels in [('all_actors_matched_refinement.jpg',body),('all_details_matched_refinement.jpg',details)]:
        combined=Image.new('RGB',(max(p.width for p in panels),sum(p.height for p in panels)),'#171717');y=0
        for panel in panels:combined.paste(panel,(0,y));y+=panel.height
        combined.save(output/filename,quality=95)
    write(output/'comparison.json',dict(CPU_only=True,CUDA_started=False,optimizer_created=False,
        metric_comparison={name:{v:m['metrics']for v,m in doc['variants'].items()}for name,doc in zip(['base7937','coarse_refinement','residual_refinement'],metrics)},
        source_pipeline_same=True,initial_prior_same=True,source_GT_masks_and_selectors_same=True,
        render_completion_sha256={str(p/'renders/complete.json'):sha(p/'renders/complete.json')for p in roots},
        base_checkpoint_sha256=spec['base_checkpoint_sha256'],matched_new_stage_Adam_reset=True,automatic_promotion=False,
        interpretation='Both refinement arms start at the same7937 learned128 function and reset Adam0/LR.03/seed42. Parameterization differs: trainable coarse128 versus frozen coarse128 plus zero256 residual. Original2064 base control reused separately. Labels/native cache/selectors/source unchanged; capped native1024 remains a physical-path limitation.'))
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();run(args.spec,args.execute)
