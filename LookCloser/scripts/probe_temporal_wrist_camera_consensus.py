"""Leave-one-fit-camera-out diagnosis; no automatic evidence deletion or mesh edit."""
from pathlib import Path
import numpy as np
from temporal_rigid_patch import fit_rigid
from joint_temporal_texture import cameras, read, sha, atomic_json
from study_wrist_observations import NAMES

ROOT=Path('/mnt/data/dec5_temporal_wrist_chain_registration')


def run():
    result=read(ROOT/'registration_result.json');data=np.load(ROOT/'correspondences.npz')
    if sha(ROOT/'correspondences.npz')!=result['array_sha256']:raise ValueError('Changed evidence')
    cameras_all,_,_=cameras('001037');lookup={r['physical_camera']:r for r in cameras_all};rows=[lookup[n] for n in NAMES]
    records=[]
    for excluded in range(5):
        fitted,center,errors=fit_rigid(data['common_gauge_points'],rows,data['point_indices'],data['camera_indices'],
                                     data['observations'],data['parameters'],[i for i in range(5) if i!=excluded])
        stats=[]
        for i,n in enumerate(NAMES):
            selected=data['camera_indices']==i
            stats.append(dict(camera=n,median=float(np.median(errors[selected])),p90=float(np.quantile(errors[selected],.9))))
        records.append(dict(excluded_camera=NAMES[excluded],parameters=fitted.tolist(),center=center.tolist(),records=stats))
    atomic_json(ROOT/'camera_consensus_diagnosis.json',dict(records=records,source_sha256=sha(ROOT/'correspondences.npz'),
        script_sha256=sha(__file__),diagnostic_only=True,automatically_accepted=False,geometry_changed=False))
    print(records,flush=True)


if __name__=='__main__':run()
