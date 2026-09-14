"""Inspect only color-qualified free-space witnesses, not every depth veto."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import numpy as np
from PIL import Image, ImageDraw
from joint_temporal_texture import read, sha, atomic_json, cameras, exr, display, ROOT
from diagnose_forearm_color_witnesses import witness_errors, patch_chroma
from study_confidence_depth_prior import project_integer
import study_forearm_plane_transfer_v3 as prior


def run(root, frame, output):
    prior.configure(); v1=prior.v2.v1
    rows, depths, depth_hashes=v1.load_real(frame)
    diagnosis=root/'free_space_diagnosis'/frame
    previous=read(diagnosis/'result.json')
    actual,_,_=cameras(frame)
    profiles=np.load(ROOT/'parameters.npz')['log_gain']
    gain=read(ROOT/'exposure.json')['fixed_exposure_gain']
    def load(pair):
        i,row=pair
        return row['physical_camera'],np.rint(display(exr(row['file_path'])*np.exp(profiles[i]),gain)*255).clip(0,255).astype(np.uint8)
    with ThreadPoolExecutor(max_workers=4) as pool:
        images=dict(pool.map(load,enumerate(actual)))
    masks=v1.masks(frame); records=[]; output.mkdir(parents=True,exist_ok=False)
    def crop(name,xy):
        x,y=np.rint(xy).astype(int)
        # Native portrait, 96x96 pixels; red cross is the exact projected point.
        im=Image.fromarray(images[name]).crop((x-48,y-48,x+48,y+48)).rotate(90)
        draw=ImageDraw.Draw(im);draw.line((43,48,53,48),fill='red');draw.line((48,43,48,53),fill='red')
        return im
    for record in previous['records']:
        name=record['camera'];path=diagnosis/(name+'.npz')
        if sha(path)!=previous['hashes'][path.name]:raise ValueError('Changed diagnosis input')
        data=np.load(path); observed=data['observed']; candidate=data['candidate']
        reference=next(r for r in rows if r['physical_camera']==name)
        errors=witness_errors(observed,reference,rows,depths,images)
        qualified=(errors<=.04).sum(0)>=3; ids=np.flatnonzero(qualified)
        summary=dict(camera=name,depth_supported=len(observed),color_qualified=len(ids),witnesses=[])
        if not len(ids):records.append(summary);continue
        # Evenly spaced query rows; deterministic and not selected for appearance.
        order=ids[np.lexsort((data['query_xy'][ids,1],data['query_xy'][ids,0]))]
        selected=order[np.linspace(0,len(order)-1,min(6,len(order))).round().astype(int)]
        panel=Image.new('RGB',(720,160*len(selected)));draw=ImageDraw.Draw(panel)
        refchroma=patch_chroma(images[name],data['query_xy'])
        for k,point in enumerate(selected):
            y=160*k;panel.paste(crop(name,data['query_xy'][point]),(0,y+32))
            draw.text((0,y),f'query {point}',fill='white');draw.text((0,y+14),name,fill='white')
            eligible=np.flatnonzero(errors[:,point]<=.04)
            best=eligible[np.argsort(errors[eligible,point])[:3]]
            sample=dict(point=int(point),query_xy=data['query_xy'][point].tolist(),sources=[])
            for j,i in enumerate(best):
                row=rows[i];uv,_=project_integer(row,observed[point:point+1]);cu,_=project_integer(row,candidate[point:point+1])
                x=120+200*j;panel.paste(crop(row['physical_camera'],uv[0]),(x,y+32))
                panel.paste(crop(row['physical_camera'],cu[0]),(x+100,y+32))
                draw.text((x,y),row['physical_camera'],fill='white')
                draw.text((x,y+14),'observed / candidate',fill='white')
                cc=patch_chroma(images[row['physical_camera']],np.rint(cu).astype(int))[0]
                ce=float(np.abs(cc-refchroma[point]).mean())
                draw.text((x,y+132),f'chroma {errors[i,point]:.4f} / {ce:.4f}',fill='white')
                sample['sources'].append(dict(camera=row['physical_camera'],observed_error=float(errors[i,point]),
                    candidate_error=ce,observed_xy=uv[0].tolist(),candidate_xy=cu[0].tolist()))
            summary['witnesses'].append(sample)
        panel.save(output/(name+'.png'))
        semantic=[]
        for row in rows:
            n=row['physical_camera']
            if n not in masks:continue
            uv,z=project_integer(row,observed[ids]);xy=np.rint(uv).astype(int)
            valid=(z>0)&(xy[:,0]>=3)&(xy[:,0]<1917)&(xy[:,1]>=3)&(xy[:,1]<1077)
            j=np.flatnonzero(valid);inside=masks[n][xy[j,1],xy[j,0]]
            semantic.append(dict(camera=n,known=int(valid.sum()),inside_forearm_roi=int(inside.sum()),outside_forearm_roi=int((~inside).sum())))
        summary['observed_projection_semantics']=semantic;records.append(summary)
    atomic_json(output/'result.json',dict(frame=frame,records=records,script_sha256=sha(__file__),
        input_diagnosis_sha256=sha(diagnosis/'result.json'),depth_hashes=depth_hashes,
        source_rgb_sha256={r['file_path']:sha(r['file_path']) for r in actual},
        profiles_sha256=sha(ROOT/'parameters.npz'),exposure_sha256=sha(ROOT/'exposure.json'),
        hashes={p.name:sha(p) for p in output.glob('*.png')},geometry_changed=False,
        candidate_rgb_comparison_not_visibility_certified=True,visual_status='pending'))
    print([{k:v for k,v in r.items() if k!='witnesses'} for r in records],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/mnt/data/dec5_forearm_annotation_domain_only'))
    p.add_argument('--frame',default='001037')
    p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_forearm_qualified_witnesses_001037'))
    a=p.parse_args();run(a.root,a.frame,a.output)
