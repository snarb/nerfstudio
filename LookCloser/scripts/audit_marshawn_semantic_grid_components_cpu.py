"""CPU-only coordinate inventory; disconnected/below-plane does not prove empty."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import hashlib
import numpy as np
from scipy import ndimage


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(campaign,output):
    import torch
    C=Path(campaign);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    candidate=C/'semantic_cell_candidates_cpu_v1/prior001_fresh2064/hard_theta01/candidate.npz'
    floor=C.parent/'two_actor_isolation/temporal_2sec_floor_v1/floor_frame.json'
    frame=json.loads(floor.read_text());ready=json.loads((C/'data/ready.json').read_text())
    if not np.array_equal(frame['world_to_box'],ready['world_to_box']):raise ValueError('Floor qualification frame differs from actual prepared coordinates')
    helper=C/'frozen_LC_semantic_physical_plan_v3/code/marshawn_frozen_semantic_head.py'
    spec=importlib.util.spec_from_file_location('_CPU_component_head',helper);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    checkpoint=C/'frozen_LC_semantic_head_archives/recovery1_20261107_v1/runs/prior001_fresh2064/head.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);head=mod.FrozenActorGrid(saved['head']['bounds'],np.eye(4),128,.01);head.load_state_dict(saved['head']);head.eval().requires_grad_(False)
    if not torch.equal(head.world_to_box,torch.eye(4)):raise ValueError('Source head must already be in floor frame')
    support_path=C/'frozen_LC_semantic_grid_replay_v2/replay/actual_CUDA_support.json';support=json.loads(support_path.read_text())
    q=C/'independent_occupancy/quarter_distortion_sourcepoint_counterfactual_v1/queries'
    observations=[p for name in ['clothing103','shoe6']for p in json.loads((q/(name+'.json')).read_text())['pixels']]
    xyz=np.asarray([p['witness_xyz']for p in observations]);assert len(xyz)==109
    report=dict(candidate_sha256=sha(candidate),floor_qualification_sha256=sha(floor),head_sha256=sha(checkpoint),head_helper_sha256=sha(helper),
        actual_CUDA_support_sha256=sha(support_path),units='prepared scene coordinates; not certified metres',source_floor_fit_max_residual=frame['max_fit_residual'],
        floor_note=frame['note'],known109_min_Z=float(xyz[:,2].min()),observations_below_minus002=int((xyz[:,2]<-.02).sum()),
        actual_CUDA_scope='26733 sampled positions; sigma values saved only for109 observations, not the isolated low cells. Global maximum is not localized.',
        actual_CUDA_maximum_raw_sigma_global=support['maximum_raw_sigma'],head_center_evaluation='CPU FP32 trilinear logits then sigmoid; not actual field sigma or GPU rendering',
        masks_modified=False,GPU_initialized=torch.cuda.is_initialized(),physical_free_space_certified=False,foot_hair_completeness_certified=False,variants={})
    with np.load(candidate)as z:
        bounds=z['cascade_aabbs'].reshape(-1,2,3);assert len(bounds)==1;b=bounds[0];pitch=(b[1]-b[0])/128
        ids=np.floor((xyz-b[0])/pitch).astype(int).clip(0,127)
        report['actual_cell_bounds']=b.tolist();report['pitch']=pitch.tolist()
        for key in ['original_active','active_r0','active_r1']:
            mask=z[key][0];core=z['source_core'][0];coords=np.argwhere(mask);centers=b[0]+(coords+.5)*pitch
            low=coords[:,2]<=1;below=(b[0,2]+(coords[:,2]+1)*pitch[2])<-.02
            with torch.inference_mode():p=head(torch.from_numpy(centers.astype(np.float32))).numpy()
            row=dict(cells=len(coords),cells_Zindex0or1=int(low.sum()),cells_entirely_below_minus002=int(below.sum()),
                center_head_p_min=float(p.min()),center_head_positive_cells=int((p>=.1).sum()),components={})
            records=[]
            for connectivity in [1,3]:
                labels,count=ndimage.label(mask,structure=ndimage.generate_binary_structure(3,connectivity));comp=[]
                for number,sl in enumerate(ndimage.find_objects(labels),start=1):
                    if sl is None:continue
                    local=np.argwhere(labels[sl]==number);offset=np.array([s.start for s in sl]);ci=local+offset
                    lo=b[0]+ci.min(0)*pitch;hi=b[0]+(ci.max(0)+1)*pitch
                    item=dict(id=number,cells=len(ci),cell_min=ci.min(0).tolist(),cell_max=ci.max(0).tolist(),bounds=[lo.tolist(),hi.tolist()],
                        source_core_cells=int(core[tuple(ci.T)].sum()),lowest_two_slices_cells=int((ci[:,2]<=1).sum()),
                        entire_component_below_minus002=bool(hi[2]<-.02),known109_observations=[int(i)for i,v in enumerate(labels[tuple(ids.T)])if v==number])
                    comp.append(item)
                comp.sort(key=lambda x:x['cells'],reverse=True);row['components'][str(6 if connectivity==1 else 26)]=comp
            for index in np.flatnonzero(low|below):
                ci=coords[index];records.append(dict(cell=ci.tolist(),lo=(b[0]+ci*pitch).tolist(),hi=(b[0]+(ci+1)*pitch).tolist(),center=centers[index].tolist(),
                    source_core=bool(core[tuple(ci)]),CPU_head_center_p=float(p[index]),actual_native_sigma_available=False))
            (out/(key+'_low_cells.json')).write_text(json.dumps(records,indent=2)+'\n');row['low_cell_inventory_sha256']=sha(out/(key+'_low_cells.json'))
            report['variants'][key]=row
    report['interpretation']='Cells near z=-.07 lie far below the fitted floor plane relative to its observed residuals; this is a diagnostic inconsistency candidate, not a local floor/visibility certificate. Sparse sole points abovezero do not prove entire feet or all underfloor volumes safe to remove. Need actual selected-density/native contribution and source-view qualification before any carving.'
    if torch.cuda.is_initialized():raise ValueError('CPU-only audit initialized CUDA')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{x:v[x]for x in ['cells','cells_Zindex0or1','cells_entirely_below_minus002']}for k,v in report['variants'].items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();audit(a.campaign,a.output)
