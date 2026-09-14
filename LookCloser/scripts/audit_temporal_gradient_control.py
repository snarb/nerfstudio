"""Independent numerical audit; continuity statistics are not GT quality scores."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from joint_temporal_texture import read, sha, atomic_json


def audit(output):
    request, receipt = read(output/'request.json'), read(output/'result.json')
    for path, digest in receipt['input_hashes'].items():
        if sha(path) != digest:
            raise ValueError('Changed correction input')
    for name, digest in receipt['hashes'].items():
        if sha(output/name) != digest:
            raise ValueError('Changed correction artifact')
    parent_paths = [Path(p) for p in receipt['input_hashes'] if Path(p).name == 'frame.png']
    if len(parent_paths) != 1:
        raise ValueError('Ambiguous source render')
    parent = parent_paths[0].parent
    x0,y0,x1,y1 = request['crop']; sl=(slice(y0,y1),slice(x0,x1))
    before=np.array(Image.open(output/'baseline.png'))
    after=np.array(Image.open(output/'corrected.png'))
    if not np.array_equal(before,np.array(Image.open(parent/'frame.png'))[sl]):
        raise ValueError('Baseline differs from published frame')
    data=np.load(output/'offset.npz'); labels=data['selection']; depth=data['depth']
    original_labels=np.rot90(np.array(Image.open(parent/'source_ids.png')))[sl].astype(np.int64)
    original_labels[original_labels==255]=-1
    if not np.array_equal(labels,original_labels):
        raise ValueError('Source labels changed')
    original_depth=np.rot90(np.load(parent/'target_depth.npz')['depth'])[sl]
    if not np.array_equal(depth,original_depth):
        raise ValueError('Depth changed')
    support=(depth>0)&(labels>=0)
    if not np.array_equal(before[~support],after[~support]):
        raise ValueError('Correction filled missing surface or background')
    offset=data['offset'].transpose(1,2,0)
    if not np.isfinite(offset).all():
        raise ValueError('Nonfinite correction')
    reconstructed=np.rint(np.clip(before/255+offset,0,1)*255).astype(np.uint8)
    if np.abs(reconstructed.astype(int)-after.astype(int)).max()>1:
        raise ValueError('Saved offset does not reproduce RGB')
    bf,af=before.astype(float)/255,after.astype(float)/255
    before_jump=[];after_jump=[];within_error=[]
    for a,b in [((slice(None),slice(None,-1)),(slice(None),slice(1,None))),
                ((slice(None,-1),slice(None)),(slice(1,None),slice(None)))]:
        dz=np.abs(np.log(np.maximum(depth[a],1e-9))-np.log(np.maximum(depth[b],1e-9)))
        connected=support[a]&support[b]&(dz<.0075)
        seam=connected&(labels[a]!=labels[b]);within=connected&~seam
        bg,ag=bf[a]-bf[b],af[a]-af[b]
        before_jump.extend(np.abs(bg[seam]).mean(-1))
        after_jump.extend(np.abs(ag[seam]).mean(-1))
        within_error.extend(np.abs(bg[within]-ag[within]).mean(-1))
    quantiles=lambda x:np.quantile(x,[.5,.9,.99]).tolist() if len(x) else None
    result=dict(scope='internal continuity and input/output invariants; not GT fidelity or temporal acceptance',
        source_labels_unchanged=True,depth_unchanged=True,unsupported_rgb_unchanged=True,
        image_quality_metrics_computed=False,source_seam_edge_count=len(before_jump),
        seam_abs_rgb_jump_median_p90_p99_before=quantiles(before_jump),
        seam_abs_rgb_jump_median_p90_p99_after=quantiles(after_jump),
        within_source_gradient_change_median_p90_p99=quantiles(within_error),
        newly_black_supported_pixels=int((support&(before.max(-1)>0)&(after.max(-1)==0)).sum()),
        corrected_supported_pixels=int((support&(before!=after).any(-1)).sum()),
        result_sha256=sha(output/'result.json'),script_sha256=sha(__file__))
    atomic_json(output/'numerical_audit.json',result)
    print(result,flush=True)


def compare_devices(cpu, cuda):
    audit(cpu); audit(cuda)
    a,b=read(cpu/'request.json'),read(cuda/'request.json')
    for key in ['frame','crop','parent_request_sha256','mesh_sha256','fixed_profiles_sha256','fixed_exposure','solver_sha256']:
        if a[key]!=b[key]:
            raise ValueError('Execution comparison changed problem inputs')
    if a.get('solver_device','cpu')!='cpu' or b['solver_device']!='cuda':
        raise ValueError('Expected CPU versus CUDA')
    ra,rb=read(cpu/'result.json'),read(cuda/'result.json')
    if ra['input_hashes']!=rb['input_hashes'] or sha(cpu/'baseline.png')!=sha(cuda/'baseline.png'):
        raise ValueError('Different source data or native baseline')
    sa,sb=ra['stats']['solver'],rb['stats']['solver']
    if any(s['dtype']!='float64' or not s['converged'] or s['max_relative_residual']>=5e-9 for s in [sa,sb]):
        raise ValueError('Unconverged execution comparison')
    if sa['ridge']!=sb['ridge']:
        raise ValueError('Changed regularization')
    x=np.array(Image.open(cpu/'corrected.png')); y=np.array(Image.open(cuda/'corrected.png'))
    error=np.abs(x.astype(int)-y.astype(int))
    delta=np.max(np.abs(np.load(cpu/'offset.npz')['offset']-np.load(cuda/'offset.npz')['offset']))
    if error.max()>1 or delta>1e-5:
        raise ValueError('CPU/CUDA difference exceeds execution tolerance')
    result=dict(frame=a['frame'],cpu_result_sha256=sha(cpu/'result.json'),cuda_result_sha256=sha(cuda/'result.json'),
        cpu_seconds=ra['elapsed_seconds'],cuda_seconds=rb['elapsed_seconds'],
        end_to_end_speedup=ra['elapsed_seconds']/rb['elapsed_seconds'],
        rgb8_max_difference=int(error.max()),changed_rgb8_channels=int((x!=y).sum()),offset_max_difference=float(delta),
        cpu_iterations=sa['iterations'],cuda_iterations=sb['iterations'],same_inputs_and_solver_equations=True,
        bit_identical=False,concurrent_host_load_may_affect_timings=True,script_sha256=sha(__file__))
    atomic_json(cuda/'device_comparison.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compare-cuda',type=Path)
    args=p.parse_args()
    if args.compare_cuda:compare_devices(args.output,args.compare_cuda)
    else:audit(args.output)
