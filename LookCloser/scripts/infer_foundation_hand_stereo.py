"""Isolated pretrained stereo inference, including reverse consistency checks.

Research-only upstream model. No training, camera optimization, held-out RGB,
synthetic images, or production mesh replacement. Checkpoint uses safe loading.
"""
import os
os.environ.setdefault('XFORMERS_DISABLED','1')
from pathlib import Path
import sys
import subprocess
import time
import numpy as np
import cv2
from PIL import Image
import torch
from omegaconf import OmegaConf
from scipy.ndimage import map_coordinates
from joint_temporal_texture import read,sha,atomic_json
from collections import defaultdict
from typing import Any
from omegaconf.base import Metadata,ContainerMetadata
from omegaconf import ListConfig,DictConfig
from omegaconf.nodes import AnyNode

ROOT=Path('/mnt/data/dec5_foundation_hand_stereo/001037')
REPO=Path('/home/brans/lookcloser_temp/FoundationStereo')
MODEL=Path('/mnt/data/dec5_foundation_model_mirror/23-51-11')


def run():
    sys.path.insert(0,str(REPO))
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder
    request=read(ROOT/'request.json');download=read(MODEL.parent/'receipt.json')
    for r in download['files']:assert sha(r['path'])==r['sha256']
    for r in request['pairs']:
        for n,h in r['hashes'].items():assert sha(Path(r['directory'])/n)==h
    output=ROOT/'inference';output.mkdir(exist_ok=False)
    configuration=OmegaConf.load(MODEL/'cfg.yaml');configuration.vit_size='vitl';configuration.valid_iters=32
    torch.set_num_threads(2);torch.manual_seed(17);np.random.seed(17)
    protocol=dict(staged_request_sha256=sha(ROOT/'request.json'),download_receipt_sha256=sha(MODEL.parent/'receipt.json'),
        model_sha256=sha(MODEL/'model_best_bp2.pth'),config_sha256=sha(MODEL/'cfg.yaml'),script_sha256=sha(__file__),
        repository_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        repository_dirty=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True),
        valid_iters=32,hierarchical=False,reverse_check='horizontal flip + swap; sample right disparity at left minus disparity',
        torch_version=torch.__version__,cuda_version=torch.version.cuda,heldout_used=False,research_only=True,
        safe_weights_only_loading=True,training_metadata_allowlist='numpy scalar/dtype; OmegaConf containers/nodes; basic Python containers',production_updated=False)
    atomic_json(output/'request.json',protocol)
    print('initializing model',flush=True);model=FoundationStereo(configuration)
    # Explicitly trusted library metadata types, never dynamically import names
    # from the checkpoint and never enable unrestricted pickle execution.
    safe=[int,dict,list,defaultdict,Any,(np._core.multiarray.scalar,'numpy.core.multiarray.scalar'),np.dtype,
          np.dtypes.Float64DType,np.dtypes.Float32DType,np.dtypes.Int64DType,
          Metadata,ContainerMetadata,ListConfig,DictConfig,AnyNode]
    with torch.serialization.safe_globals(safe):
        checkpoint=torch.load(MODEL/'model_best_bp2.pth',map_location='cpu',weights_only=True,mmap=True)
    status=model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
    assert not status.missing_keys and not status.unexpected_keys
    model.cuda().eval();print('strict checkpoint loaded',flush=True)
    def predict(left,right):
        a=torch.from_numpy(left.copy()).permute(2,0,1)[None].float().cuda()
        b=torch.from_numpy(right.copy()).permute(2,0,1)[None].float().cuda()
        padder=InputPadder(a.shape,divis_by=32,force_square=False);a,b=padder.pad(a,b)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            disparity=model.forward(a,b,iters=32,test_mode=True)
        return padder.unpad(disparity.float())[0,0].cpu().numpy()
    results=[]
    for record in request['pairs']:
        source=Path(record['directory']);dest=output/source.name;dest.mkdir();start=time.time()
        left=np.array(Image.open(source/'left.png'));right=np.array(Image.open(source/'right.png'))
        dl=predict(left,right);print('forward',source.name,flush=True)
        dr=predict(right[:,::-1],left[:,::-1])[:,::-1].copy();print('reverse',source.name,flush=True)
        if not np.isfinite(dl).all() or not np.isfinite(dr).all():raise ValueError('Nonfinite model output')
        yy,xx=np.indices(dl.shape,dtype=np.float32);xr=xx-dl
        rdisp=map_coordinates(dr,[yy,xr],order=1,mode='constant',cval=-1000)
        error=np.abs(dl-rdisp)
        cal=np.load(source/'calibration.npz')
        def observed(prefix):
            x=cal[prefix+'_map_x'];y=cal[prefix+'_map_y']
            return (x>=1)&(x<1078)&(y>=1)&(y<1918)
        right_observed=map_coordinates(observed('right').astype(np.float32),[yy,xr],order=1,mode='constant',cval=0)>.9999
        common=observed('left')&right_observed&(dl>0)&(xr>=0)&(xr<=dl.shape[1]-1)
        metric_disp=dl-float(cal['disparity_offset'])
        common &= metric_disp>0
        depth=np.zeros_like(dl);depth[common]=cal['cropped_intrinsic'][0,0]*cal['baseline']/metric_disp[common]
        np.savez_compressed(dest/'prediction.npz',left_disparity=dl,right_disparity=dr,lr_error=error,
                            valid_source_domain=common,rectified_depth=depth,consistent=common&(error<=2))
        # Display diagnostics, never substitute colormap for RGB prediction.
        color=cv2.applyColorMap(np.rint(np.clip(dl/256,0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)[...,::-1]
        color[~common]=0;Image.fromarray(np.concatenate([left,color],axis=1)).save(dest/'disparity_review.png')
        skin=cal['left_mask'].astype(bool)
        result=dict(pair=source.name,seconds=time.time()-start,finite=True,shape=list(dl.shape),
            skin_mask_pixels=int(skin.sum()),skin_common_domain=int((skin&common).sum()),
            skin_lr_consistent={str(t):int((skin&common&(error<=t)).sum()) for t in [1,2,3]},
            prediction_sha256=sha(dest/'prediction.npz'),review_sha256=sha(dest/'disparity_review.png'),
            learned_depth_not_measured=True,geometry_changed=False)
        atomic_json(dest/'complete.json',result);results.append(result);print(result,flush=True)
    hub=Path(torch.hub.get_dir())/'facebookresearch_dinov2_main'
    atomic_json(output/'complete.json',dict(request_sha256=sha(output/'request.json'),pairs=results,
        dinov2_source_hashes={str(p):sha(p) for p in hub.rglob('*.py')} if hub.exists() else {},visual_status='pending'))


if __name__=='__main__':run()
