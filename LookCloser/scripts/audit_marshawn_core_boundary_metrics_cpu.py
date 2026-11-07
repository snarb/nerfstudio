"""Fixed-region PNG diagnostic only. No V1 reranking, GPU or changed labels."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def regions(mask, radius=8):
    if mask.dtype!=bool or mask.ndim!=2 or radius<=0:raise ValueError('Boolean2D mask and positive radius required')
    distance=distance_transform_edt(np.pad(mask,1,constant_values=False))[1:-1,1:-1]
    core=mask&(distance>radius);band=mask&~core
    assert not (core&band).any() and np.array_equal(core|band,mask)
    return {'whole_PNG_control':mask,'core_gt8px':core,'boundary_le8px':band}


def main(c,out):
    import torch,torchmetrics
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from marshawn_tuning_metrics import masked_actor_metrics
    assert not torch.cuda.is_initialized();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();base=c/'independent_occupancy/quarter_distortion_source_margin_v1/quarter_distortion_raw126316';arms={'A':base/'native_A/renders/step-000126316','B0':c/'independent_occupancy/quarter_distortion_sourcepoint_counterfactual_v1/quarter_distortion_raw126316/source_point_B/renders/step-000126316','.001':base/'margin001/renders/step-000126316','.002':base/'margin002/renders/step-000126316'}
    data=c/'data';ready=json.load(open(data/'ready.json'));rows={r['camera']:r for r in ready['records']};model=LearnedPerceptualImagePatchSimilarity(normalize=True).net.eval().cpu();assert model.pnet_type=='alex' and model.resize is None
    inputs={str(data/'ready.json'):sha(data/'ready.json')};results=[];mask_receipts=[]
    for cam in ['DC','HC','KA']:
        targetpath=data/rows[cam]['image'];target=torch.from_numpy(np.array(Image.open(targetpath).convert('RGB'),dtype=np.float32)/255).permute(2,0,1)[None];inputs[str(targetpath)]=sha(targetpath);maskpath=data/'semantics'/f'{cam}.npz';inputs[str(maskpath)]=sha(maskpath)
        with np.load(maskpath)as z:masks={actor:regions(z[actor])for actor in ['black','grey']}
        for actor,rs in masks.items():
            for region,mask in rs.items():
                if not mask.any():raise ValueError('Empty diagnostic region')
                mask_receipts.append(dict(camera=cam,actor=actor,region=region,pixels=int(mask.sum()),mask_array_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),crop_xyxy=rows[cam]['crops'][actor]))
        for arm,path in arms.items():
            rgbpath=path/f'{cam}.png';alphapath=path/f'{cam}_alpha.png';inputs[str(rgbpath)]=sha(rgbpath);inputs[str(alphapath)]=sha(alphapath);pred=torch.from_numpy(np.array(Image.open(rgbpath).convert('RGB'),dtype=np.float32)/255).permute(2,0,1)[None]
            for actor,rs in masks.items():
                for region,mask in rs.items():
                    with torch.inference_mode():r=masked_actor_metrics(pred,target,torch.from_numpy(mask.copy())[None,None],model,crop_xyxy=rows[cam]['crops'][actor])
                    results.append(dict(camera=cam,actor=actor,region=region,arm=arm,psnr=r['psnr'],ssim=r['ssim'],lpips=r['lpips'],pixels=int(mask.sum()),crop_xyxy=r['crop_xyxy']))
            (out/'progress.json').write_text(json.dumps({'camera':cam,'arm':arm,'records':len(results),'seconds':time.monotonic()-start}))
            print(json.dumps({'camera':cam,'arm':arm,'records':len(results),'seconds':round(time.monotonic()-start,1)}),flush=True)
    aggregate={}
    for region in ['whole_PNG_control','core_gt8px','boundary_le8px']:
        aggregate[region]={}
        for arm in arms:
            selected=[r for r in results if r['region']==region and r['arm']==arm];assert len(selected)==6
            aggregate[region][arm]={k:float(np.mean([r[k]for r in selected]))for k in ['psnr','ssim','lpips']}
    assert not torch.cuda.is_initialized()
    checkpoint=Path.home()/'.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth'
    receipt=dict(kind='FIXED_EUCLIDEAN_CORE_BOUNDARY_PNG_CPU_DIAGNOSTIC_NOT_LEADERBOARD',radius_HD_pixels=8,erosion='Euclidean EDT>8; padded image exterior false; boundary originalmask minuscore',aggregation='equal2actors within each of3views, thenequalviews',LPIPS='sameAlex/noresize/normalizeTrue, identicalRGB.5exterior, originalsealedcrop fixedacrossallregionsandarms',records=results,aggregate=aggregate,mask_receipts=mask_receipts,inputs_sha256=inputs,helper_sha256=sha(__file__),metric_helper_sha256=sha(Path(__file__).with_name('marshawn_tuning_metrics.py')),alexnet_weights_sha256=sha(checkpoint),torch_version=torch.__version__,torchmetrics_version=torchmetrics.__version__,seconds=time.monotonic()-start,CUDA_initialized=False,model_or_labels_modified=False,V1leaderboard_unchanged=True,limitations=['PNG8 quantization differs from savedfloat V1 metrics; allarms use identical PNGdomain.','BoundaryLPIPS averages fixedcrop including extensive identicalneutralexterior; comparearmswithinregion, notabsolutes acrossregions.','Erodedsemanticmask is not physical visibility or optical alpha groundtruth.'])
    (out/'metrics.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(aggregate,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.campaign,a.output)
