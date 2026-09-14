"""Read-only contrastive decisions on previously inspected same-time witnesses."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from joint_temporal_texture import read,sha,atomic_json,cameras,exr,display,ROOT
from contrastive_forearm_witnesses import comparison_votes
import study_forearm_plane_transfer_v3 as prior


def run():
    frame='001037';prior.configure();v1=prior.v2.v1;rows,depths,hashes=v1.load_real(frame)
    actual,_,_=cameras(frame);profiles=np.load(ROOT/'parameters.npz')['log_gain'];gain=read(ROOT/'exposure.json')['fixed_exposure_gain']
    def load(pair):
        i,r=pair
        return r['physical_camera'],np.rint(display(exr(r['file_path'])*np.exp(profiles[i]),gain)*255).clip(0,255).astype(np.uint8)
    with ThreadPoolExecutor(max_workers=4) as pool:images=dict(pool.map(load,enumerate(actual)))
    source=Path('/mnt/data/dec5_forearm_annotation_domain_only/free_space_diagnosis')/frame
    visual=read('/mnt/data/dec5_forearm_qualified_witnesses_001037/result.json')
    output=Path('/mnt/data/dec5_forearm_contrastive_witness_control');output.mkdir(exist_ok=False)
    records=[];arrays={}
    for row in visual['records']:
        name=row['camera'];data=np.load(source/(name+'.npz'));camera=next(r for r in rows if r['physical_camera']==name)
        votes=comparison_votes(data['observed'],data['candidate'],camera,rows,depths,images)
        for k,v in votes.items():arrays[name+'_'+k]=v
        selected=[r['point'] for r in row['witnesses']]
        records.append(dict(camera=name,previous_rgb_qualified=int((votes['rgb_qualified']>=3).sum()),
            comparison_qualified=int((votes['decisive']>=3).sum()),
            previously_inspected_samples=[dict(point=i,**{k:int(v[i]) for k,v in votes.items()}) for i in selected]))
    np.savez_compressed(output/'votes.npz',**arrays)
    atomic_json(output/'result.json',dict(frame=frame,records=records,script_sha256=sha(__file__),
        helper_sha256=sha(Path(__file__).with_name('contrastive_forearm_witnesses.py')),margin=.01,
        source_diagnosis_sha256=sha(source/'result.json'),previous_visual_sha256=sha('/mnt/data/dec5_forearm_qualified_witnesses_001037/visual_review.json'),
        source_rgb_sha256={r['file_path']:sha(r['file_path']) for r in actual},source_depth_hashes=hashes,
        profiles_sha256=sha(ROOT/'parameters.npz'),exposure_sha256=sha(ROOT/'exposure.json'),
        votes_sha256=sha(output/'votes.npz'),geometry_changed=False,not_a_general_confidence_calibration=True))
    print(records,flush=True)


if __name__=='__main__':run()
