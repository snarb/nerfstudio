"""Common v1/v2 actor metrics for a frozen native semantic-support candidate."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation,binary_erosion
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from marshawn_tuning_metrics import masked_actor_metrics,aggregate_actor_metrics


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    data=a.base.parent/'data';ready=json.loads((data/'ready.json').read_text());records={r['camera']:r for r in ready['records']}
    path=a.base.parent/'eval_annotation_v2_SOURCE_ONLY/eval_v2_manifest.json';v2=json.loads(path.read_text());assert v2['training_use_forbidden'] and not v2['model_renders_used'];v2rows={r['camera']:r for r in v2['records']}
    variants={'native_clip20k':'inside','partial_occlusion_gate0':'raw_semantic_gate0','partial_occlusion_pad1':'raw_semantic_gate_pad1','qualified_min2_gate0':'semantic_gate0','qualified_min2_pad1':'semantic_gate_pad1'}
    rows={v:{k:[] for k in variants} for v in ['v1','v2']};diagnostics=[];torch.set_num_threads(2);lp=LearnedPerceptualImagePatchSimilarity(normalize=True).cuda().eval().net.eval()
    for camera in ['DC','HC','KA']:
        r=records[camera];assert sha(data/r['image'])==r['rgb_sha256'] and sha(r['semantic'])==r['semantic_sha256'] and sha(v2rows[camera]['v2_archive'])==v2rows[camera]['v2_sha256']
        source=torch.tensor(np.asarray(Image.open(data/r['image']).convert('RGB'),np.float32)/255,device='cuda').permute(2,0,1)[None]
        with np.load(a.output/'renders'/camera/'raw_partition.npz') as render:
            for version,mp in [('v1',r['semantic']),('v2',v2rows[camera]['v2_archive'])]:
                with np.load(mp) as masks:
                    union=masks['black']|masks['grey'];outside=~binary_dilation(union,iterations=3)
                    for variant,key in variants.items():
                        rgba=render[key];pred=torch.tensor(rgba[...,:3],device='cuda').permute(2,0,1)[None]
                        for actor in ['black','grey']:
                            mask=masks[actor];metric=masked_actor_metrics(pred,source,torch.tensor(mask,device='cuda')[None,None],lp,crop_xyxy=r['crops'][actor]);rows[version][variant].append(dict(view=camera,actor=actor,**metric))
                        if version=='v2':
                            edge=union&~binary_erosion(union,iterations=3);delta=rgba-render['raw_semantic_gate_pad1']
                            diagnostics.append(dict(view=camera,variant=variant,outside_v2_dilation3_alpha_gt01=int(((rgba[...,3]>.1)&outside).sum()),actor_alpha_drop_gt01_vs_partial_pad1=int((delta[...,3][union]<-.1).sum()),edge_RGB_rmse_vs_partial_pad1=float(np.sqrt(np.mean(delta[...,:3][edge]**2)))))
    result=dict(results={v:{k:aggregate_actor_metrics(x) for k,x in values.items()} for v,values in rows.items()},diagnostics=diagnostics,receipt=dict(ready_sha256=sha(data/'ready.json'),v2_manifest_sha256=sha(path),metric_helper_sha256=sha(Path(__file__).with_name('marshawn_tuning_metrics.py')),semantic_grid_sha256=sha(a.output/'semantic_grid.npz'),no_eval_input_to_gate=True,domain='Original fixed actor crops, neutral0.5 exterior, fullHD floatRGB, equal actors/views',variants=variants))
    (a.output/'metrics_v1_v2.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({v:{k:{m:y for m,y in x['metrics'].items() if m!='validation_loss'} for k,x in values.items()} for v,values in result['results'].items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
