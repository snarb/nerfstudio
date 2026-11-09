"""Actual retained-point TRAIN node support/current class pressure; CPU only."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from marshawn_frozen_semantic_head import FrozenActorGrid,rendered_classes,tensor_digest
from audit_marshawn_semantic_residual_exposure_cpu import nodes
from prepare_marshawn_frozen_semantic_cpu import sha,write


def run(c,physical,output):
    torch.set_num_threads(2)
    output.mkdir(parents=True,exist_ok=False)
    trace=physical/'prior001/renders/DC_trace_1097_572.npz'
    capture_path=c/'frozen_LC_semantic_capture_plan_v4/capture/complete.json';capture=json.loads(capture_path.read_text())
    contract=json.loads((c/'frozen_LC_semantic_head_plan_v1/head_input_contract.json').read_text())
    completion=json.loads((physical/'heads/prior001_local_completion_adapter.json').read_text());checkpoint=Path(completion['checkpoint'])
    if sha(checkpoint)!=completion['checkpoint_sha256']:raise ValueError('Actual head changed')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    head=FrozenActorGrid(contract['bounds'],np.eye(4),128,initial_probability=.01);head.load_state_dict(saved['head'],strict=True);head.eval();head.requires_grad_(False)
    digest=tensor_digest(head)
    with np.load(trace)as z:
        i=int(np.argmax(z['hard_theta05_selected_weight']));point=z['xyz'][i].copy()
        actual=dict(sample_index=i,XYZ=point.tolist(),selected_weight=float(z['hard_theta05_selected_weight'][i]),native_weight=float(z['weight'][i]),p_GPU=float(z['p'][i]),sigma=float(z['sigma'][i]))
    bounds=head.bounds.numpy();kernel=np.zeros(128**3,np.float64);vertices=[]
    logits=head.logits.detach().numpy().reshape(-1)
    for index,basis in nodes(point[None],bounds):
        flat=int(index[0]);kernel[flat]+=float(basis[0]);vertices.append(dict(flat_index=flat,XYZ_node=[flat//128**2,(flat//128)%128,flat%128],
            basis=float(basis[0]),trained_logit=float(logits[flat]),trained_node_probability=float(1/(1+np.exp(-logits[flat])))))
    all_rows=[];global_offset=0;positive_pressure=0.;negative_pressure=0.;hashes={str(p):sha(p)for p in [trace,capture_path,checkpoint]}
    for row in capture['records']:
        if not row['rays']:continue
        archive=Path(row['archive'])
        if sha(archive)!=row['archive_sha256']:raise ValueError('Source visibility changed')
        hashes[str(archive)]=row['archive_sha256']
        with np.load(archive)as z:
            data={k:z[k].copy()for k in ['xyz','weight','ray_indices','semantic_target','label_weight','pixel_xy','category']}
        xyz,w,ids=data['xyz'],data['weight'],data['ray_indices'].astype(int);n=row['rays']
        with torch.no_grad():
            p=head(torch.from_numpy(xyz)).numpy()
            classes=rendered_classes(torch.from_numpy(p),torch.from_numpy(w),torch.from_numpy(ids),n).numpy()
        alpha=classes[:,:2].sum(-1);q=classes[:,0]/np.maximum(alpha,1e-12)
        inside=((xyz>=bounds[0])&(xyz<=bounds[1])).all(-1)
        overlap=np.zeros(len(xyz),np.float64)
        for index,basis in nodes(xyz,bounds):overlap+=basis*kernel[index]
        overlap*=inside
        exposure=np.bincount(ids,weights=w*overlap,minlength=n)
        projected=np.bincount(ids,weights=w*p*(1-p)*overlap,minlength=n)
        confidence=data['label_weight'];target=data['semantic_target']
        multiplier=(1-target)/np.maximum(1-q,1e-7)-target/np.maximum(q,1e-7)
        multiplier[(q<=1e-7)|(q>=1-1e-7)]=0
        gradient=confidence*projected*multiplier/n/60
        near=np.full(n,np.inf)
        selected=(overlap>0)&(w>1e-8)&(confidence[ids]>0)
        np.minimum.at(near,ids[selected],np.linalg.norm(xyz[selected]-point,axis=1))
        for query in np.flatnonzero((exposure>1e-12)&(confidence>0)):
            record=dict(camera=row['camera'],camera_ray_index=int(query),global_query_id=global_offset+int(query),pixel_xy=data['pixel_xy'][query].tolist(),
                semantic_target=float(target[query]),category=str(data['category'][query]),native_alpha=float(alpha[query]),current_Qactor_over_native_alpha=float(q[query]),
                actual_point_shared_node_native_visibility_exposure=float(exposure[query]),
                current_expected_objective_projected_logit_gradient=float(gradient[query]),
                nearest_shared_node_sample_distance_with_nativew_gt1e_minus8=None if not np.isfinite(near[query])else float(near[query]))
            all_rows.append(record)
            if target[query]==1:positive_pressure+=float(gradient[query])
            else:negative_pressure+=float(gradient[query])
        global_offset+=n
    if global_offset!=69888 or tensor_digest(head)!=digest:raise ValueError('Query/head immutability failed')
    positives=sorted([r for r in all_rows if r['semantic_target']==1],key=lambda r:-abs(r['current_expected_objective_projected_logit_gradient']))
    negatives=sorted([r for r in all_rows if r['semantic_target']==0],key=lambda r:-abs(r['current_expected_objective_projected_logit_gradient']))
    frames={r['physical_camera']:r for r in json.loads((c/'data/transforms.json').read_text())['frames']}
    for label,records in [('positive',positives),('negative',negatives)]:
        panel=Image.new('RGB',(1400,540),'#171717');draw=ImageDraw.Draw(panel)
        draw.text((4,4),'ACTUAL DC[1097,572] dominant selectedXYZ shared-node '+label+' TRAIN support; no label changes',fill='white')
        for k,record in enumerate(records[:10]):
            frame=frames[record['camera']];path=c/'data'/frame['file_path'];hashes[str(path)]=sha(path);x,y=record['pixel_xy']
            tile=Image.open(path).convert('RGB').crop((x-24,y-24,x+25,y+25)).resize((196,196),Image.Resampling.NEAREST)
            ImageDraw.Draw(tile).rectangle((94,94,101,101),outline='red')
            xx,yy=(k%5)*280,30+(k//5)*255;panel.paste(tile,(xx,yy))
            draw.text((xx,yy+199),record['camera']+' '+str(record['pixel_xy'])+' id'+str(record['global_query_id']),fill='white')
            draw.text((xx,yy+215),'w*b exposure %.5g'%record['actual_point_shared_node_native_visibility_exposure'],fill='white')
            draw.text((xx,yy+231),'current grad %.5g'%record['current_expected_objective_projected_logit_gradient'],fill='white')
        panel.save(output/(label+'_TRAIN_actual_point_sources.png'))
    with(output/'all_shared_node_queries.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0])if all_rows else[]);writer.writeheader();writer.writerows(all_rows)
    write(output/'actual_point_support.json',dict(CPU_only=True,CUDA_initialized=torch.cuda.is_initialized(),optimizer_created=False,
        head_parameters_unchanged=True,actual_physical_trace_point=actual,point_head_CPU_probability=float(head(torch.from_numpy(point[None]))[0]),
        interpolation_nodes=vertices,positive_query_count=len(positives),negative_query_count=len(negatives),
        current_expected_objective_positive_projected_gradient=positive_pressure,current_expected_objective_negative_projected_gradient=negative_pressure,
        current_expected_objective_net_projected_gradient=positive_pressure+negative_pressure,
        top20_positive_queries=positives[:20],top20_negative_queries=negatives[:20],source_sha256=hashes,
        semantics='Restricts interpolation support to one actual dominant selectedσ/weight XYZ, not a whole EVAL classifier ray. Exposure is sum(nativew*b_train*b_point); current class-pressure projects the fixed conditional-CE gradient onto this point logit basis, expected uniform60-camera sampling with eachpacket fixed N normalization. Negative gradient would favor higher actor logit under a plain gradient step; Adam/momentum and past trajectory are NOT inferred. Basis arithmetic is a CPU analytical diagnostic, not a new training update. Source photos require qualification; no EVALfitlabel, whole-ray-empty or direct hiddenpoint classification claim.'))
    print(json.dumps(dict(positive_queries=len(positives),negative_queries=len(negatives),positive_gradient=positive_pressure,negative_gradient=negative_pressure,net=positive_pressure+negative_pressure)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--physical-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.campaign,args.physical_root,args.output)
