"""Independent double-precision triangle/ray check of residual attribution."""
from pathlib import Path
import argparse
import numpy as np
from run_local_mhr_completion import read,save,sha,require


def intersections(ray,vertices,triangles):
    """Two-sided Moller-Trumbore; return original triangle IDs and positive t."""
    tv=np.asarray(vertices,np.float64)[triangles];e1=tv[:,1]-tv[:,0];e2=tv[:,2]-tv[:,0]
    origin,direction=np.asarray(ray,np.float64).reshape(2,3)
    p=np.cross(direction,e2);det=np.einsum('ij,ij->i',e1,p);valid=abs(det)>1e-15
    inv=np.zeros_like(det);inv[valid]=1/det[valid];delta=origin-tv[:,0]
    u=np.einsum('ij,ij->i',delta,p)*inv;q=np.cross(delta,e1)
    v=(q@direction)*inv;t=np.einsum('ij,ij->i',e2,q)*inv
    keep=valid&(u>=0)&(v>=0)&(u+v<=1)&(t>0)
    return np.flatnonzero(keep),t[keep]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--tests',type=Path,required=True);a=p.parse_args()
    root=a.output;require(not (root/'final_seal.json').exists(),'Already sealed')
    result=read(root/'result.json');checked={}
    for path,h in result['input_hashes'].items():require(sha(path)==h,'Changed source');checked[path]=h
    for name,h in result['outputs'].items():require(sha(root/name)==h,'Changed diagnostic output')
    producer=Path(__file__).with_name('diagnose_local_mhr_residual_rays.py');require(sha(producer)==result['script_sha256'],'Changed producer')
    for name,h in result['helpers'].items():require(sha(producer.with_name(name))==h,'Changed helper')
    prior_path=next(Path(p) for p in checked if p.endswith('/silhouette100/fit.npz') and '/dec5_mhr_transfer_001083/' in p)
    f=np.load(prior_path);e=np.load(root/'rays.npz');replay=[]
    # This bounded case has no full-prior hits. Double precision independently
    # confirms absence; later stages are subsets/subdivisions of that prior.
    for i,row in enumerate(result['records']):
        ids,t=intersections(e['rays'][i],f['vertices'],f['triangles'])
        require(len(ids)==0,'Full-prior miss not independently reproduced')
        require(all(not q['hit'] and q['scene_triangle_id'] is None for q in row['stages'].values()),'Unexpected stage hit')
        for name in row['stages']:
            require(np.isinf(e[name+'_depth'][i]),'Saved hit differs')
            require(int(e[name+'_triangle_local'][i])==np.iinfo(np.uint32).max,'Wrong no-hit sentinel')
        require(row['first_absent_stage']=='prior_full','Wrong bottleneck')
        require(row['ray_origin']+row['ray_direction']==e['rays'][i].tolist(),'Ray receipt differs')
        x,y=row['portrait_xy'];require(row['landscape_xy']==[1919-y,x],'Rotation mismatch')
        replay.append(dict(portrait_xy=row['portrait_xy'],double_precision_full_prior_intersections=0))
    visual=read(root/'visual_review.json');require(visual['reviewer']=='LLM' and not visual['production_changed'],'Review missing')
    for path,h in visual['viewed_images'].items():require(sha(path)==h,'Reviewed image changed');checked[path]=h
    for path,h in visual.get('reference_inputs',{}).items():require(sha(path)==h,'Reference evidence changed');checked[path]=h
    for path in [producer,Path(__file__),a.report,a.tests]:checked[str(path.resolve())]=sha(path)
    save(root/'final_seal.json',dict(status='passed',checked_bindings=checked,
        inventory={str(f.relative_to(root)):sha(f) for f in sorted(root.rglob('*')) if f.is_file()},
        independent_ray_triangle_checks=replay,stage_records_checked=sum(len(r['stages']) for r in result['records']),
        no_geometry_mutation=True,no_fit=True,scope='Three residual rays; not a dense under-cheek repair evaluation'))
    print('Passed',len(checked),'bindings',len(replay),'independent full-prior rays',flush=True)


if __name__=='__main__':main()
