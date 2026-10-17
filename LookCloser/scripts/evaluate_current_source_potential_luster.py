"""Evaluation-only source counterfactuals; never constructs a GT-selected image.

Each prediction is made without GT from one fixed TRAIN camera and frozen LC.
Only scalar errors are compared afterwards. Tile minima are optimistic
diagnostics with incompatible neighborhoods, NOT achievable hybrid metrics or
a rigorous lower bound. No minimizing source labels or RGB mosaic are saved.
"""
import argparse,subprocess,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from luster_hybrid_common import bootstrap,geometry,read,write,sha,assert_source,native_rows
from supported_luster_hybrid import geometric_support
from seam_single_luster_hybrid import merge


def counterfactual(base,record,source,config):
    """GT-free predictor; every admitted pixel uses this one native photo."""
    valid=geometric_support(record,config)
    labels=np.where(valid,source,-1).astype(np.int16)
    # Same island removal and per-source merge as the current candidate family.
    labels[valid&(cv2.medianBlur(valid.astype(np.uint8),3)==0)]=-1
    result,weight=merge(base,[(source,record)],labels,{**config,'filter_support':'geometry','adaptive_seam':True,'seam_fade':3.,'seam_tolerance':.02})
    assert np.array_equal(result[weight==0],base[weight==0])
    return result


def tile_errors(values,mask,size):
    errors=[];pixels=[]
    for y in range(0,mask.shape[0],size):
        for x in range(0,mask.shape[1],size):
            sl=np.s_[y:y+size,x:x+size];m=mask[sl]
            if m.any():errors.append(float(values[sl][m].sum()));pixels.append(int(m.sum()))
    return np.array(errors),np.array(pixels)


def run(root,parent,wait_pids):
    deadline=read(root/'preflight/manifest.json')['deadline']
    while any(Path(f'/proc/{pid}').exists() for pid in wait_pids):
        if time.time()>deadline:raise TimeoutError('Campaign deadline')
        time.sleep(3)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU in use'
    bootstrap(root)
    import lpips
    torch.set_num_threads(4);cv2.setNumThreads(4)
    metric=lpips.LPIPS(net='alex',spatial=True).eval().cuda()
    protocol=read(root/'evaluation/protocol.json');old=root/'candidates'/parent;config=read(old/'config.json');native=native_rows()
    out=root/'evaluation'/f'{parent}_current_source_potential';out.mkdir(exist_ok=True)
    rows=[];sizes=[32,64,128,256];reference=read(root/'evaluation/baseline_metrics.json')
    for target in [7,12,95,150]:
        if time.time()>deadline:raise TimeoutError('Campaign deadline')
        receipt=read(old/f'cam{target:03d}_receipt.json');warp_root=Path(receipt.get('warp_root',root))
        first=warp_root/'warps'/f'cam{target:03d}'/f"src{receipt['sources'][0]:03d}"
        bx0,by0,bx1,by1=read(first/'receipt.json')['box'];prediction_slice=np.s_[by0:by1,bx0:bx1]
        base=geometry(root,target)['rgb'];selected=np.load(old/f'cam{target:03d}.npy',mmap_mode='r')
        e=protocol['cameras'][str(target)]
        assert sha(e['source'])==e['source_sha256'] and sha(e['mask'])==e['mask_sha256']
        gt=np.asarray(Image.open(e['source']).convert('RGB'),np.float32)/255;mask=np.asarray(Image.open(e['mask']))>0
        yy,xx=np.where(mask);x0=max(0,int(xx.min())-16);x1=min(mask.shape[1],int(xx.max())+17);y0=max(0,int(yy.min())-16);y1=min(mask.shape[0],int(yy.max())+17)
        evaluation_slice=np.s_[y0:y1,x0:x1];m=mask[evaluation_slice]
        mt=torch.from_numpy(m.astype(np.float32))[None,None].cuda()
        gt_tensor=torch.from_numpy(np.array(gt[evaluation_slice])).permute(2,0,1)[None].cuda();gt_tensor=gt_tensor*mt+.5*(1-mt)
        def evaluate(prediction):
            with torch.inference_mode():
                p=torch.from_numpy(np.array(prediction[evaluation_slice])).permute(2,0,1)[None].cuda();p=p*mt+.5*(1-mt)
                return metric(p*2-1,gt_tensor*2-1)[0,0].cpu().numpy()
        baseline_map=evaluate(base);selected_map=evaluate(selected)
        baseline=float(baseline_map[m].mean());current=float(selected_map[m].mean())
        expected=next(r['lpips'] for r in reference['rows'] if r['camera']==target and r['region']=='person')
        assert abs(baseline-expected)<1e-6
        minimum=np.minimum(baseline_map,selected_map)
        best_tiles={size:np.minimum(tile_errors(baseline_map,m,size)[0],tile_errors(selected_map,m,size)[0]) for size in sizes}
        inputs={};source_rows=[]
        for source in receipt['sources']:
            if time.time()>deadline:raise TimeoutError('Campaign deadline')
            assert_source(native,target,source)
            directory=warp_root/'warps'/f'cam{target:03d}'/f'src{source:03d}';r=read(directory/'receipt.json')
            assert r['box']==[bx0,by0,bx1,by1]
            record={key:np.load(directory/f'{key}.npy',mmap_mode='r') for key in r['arrays']}
            # Construction uses only prediction-derived support, no GT crop.
            prediction=np.array(base);prediction[prediction_slice]=counterfactual(np.asarray(base[prediction_slice]),record,source,config)
            errors=evaluate(prediction);minimum=np.minimum(minimum,errors)
            for size in sizes:best_tiles[size]=np.minimum(best_tiles[size],tile_errors(errors,m,size)[0])
            source_rows.append(dict(source=source,person_lpips=float(errors[m].mean())))
            inputs[str(directory/'receipt.json')]=sha(directory/'receipt.json')
            print(dict(camera=target,source=source,person_lpips=source_rows[-1]['person_lpips']),flush=True)
        row=dict(camera=target,baseline=baseline,current=current,counterfactuals=source_rows,
                 optimistic_pixel_min=float(minimum[m].mean()),optimistic_tile_min={str(size):float(best_tiles[size].sum()/m.sum()) for size in sizes},input_receipts=inputs)
        rows.append(row);write(out/f'cam{target:03d}.json',row)
    mean=lambda key:float(np.mean([r[key] for r in rows if r['camera'] in [12,95,150]]))
    summary=dict(usage='evaluation_only_never_construction',parent=parent,rows=rows,baseline_mean=mean('baseline'),current_mean=mean('current'),
                 optimistic_pixel_min_mean=mean('optimistic_pixel_min'),optimistic_tile_min_mean={str(size):float(np.mean([r['optimistic_tile_min'][str(size)] for r in rows if r['camera'] in [12,95,150]])) for size in sizes},
                 code_sha256=sha(Path(__file__)),merge_code_sha256=sha(Path(__file__).with_name('seam_single_luster_hybrid.py')),protocol_sha256=sha(root/'evaluation/protocol.json'),
                 warning='Minima are evaluation-only comparisons of incompatible image neighborhoods, not final hybrid metrics, achievable gains, or a rigorous lower bound. No GT-selected texture image or minimizing label map is created.')
    write(out/'summary.json',summary);print({k:v for k,v in summary.items() if k not in ['rows']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',default='single_source_contradiction400_once_seam');p.add_argument('--wait-pids',nargs='+',type=int,default=[]);a=p.parse_args();run(a.root,a.parent,a.wait_pids)
