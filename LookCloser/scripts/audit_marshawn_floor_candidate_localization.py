"""Fixed-camera localization sensitivity for reviewed TRAIN floor candidates; no plane fit."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from audit_marshawn_geometry import projection


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(a):
    review=json.loads(a.review.read_text());tf=json.loads(a.transforms.read_text());rows={f['physical_camera']:f for f in tf['frames']if f['file_path']in tf['train_filenames']}
    assert len(rows)==63 and not set(rows)&{'DC','HC','KA'}
    t=np.array(json.loads(a.box.read_text())['world_to_box']);inverse=np.linalg.inv(t);result=[]
    for row in review['records']:
        if not row['localized_floor_material_candidate']:continue
        graph=row['triangulation'];xyz=np.array(graph['xyz_original']);jacobians=[]
        assert np.allclose(t[:3,:3]@xyz+t[:3,3],row['xyz_floor'],atol=1e-12,rtol=0)
        for ob in graph['observations']:
            matrix,*_=projection(rows[ob['camera']]);q=matrix@np.r_[xyz,1.];assert q[2]>0
            jac=(matrix[:2,:3]*q[2]-q[:2,None]*matrix[2,:3])/q[2]**2
            jacobians.extend(jac@inverse[:3,:3])
        jacobians=np.array(jacobians);covariance=.5**2*np.linalg.inv(jacobians.T@jacobians)
        result.append(dict(source_point_id=row['source_point_id'],xyz_floor=row['xyz_floor'],sigma_HD_pixel_assumed=.5,
                           covariance_floor=covariance.tolist(),three_sigma_Z=float(3*np.sqrt(covariance[2,2])),
                           three_sigma_major_axis=float(3*np.sqrt(np.linalg.eigvalsh(covariance)[-1])),all_source_observations=graph['observations']))
    report=dict(records=result,source_bindings={str(p):sha(p)for p in[a.review,a.transforms,a.box,Path(__file__),Path(__file__).with_name('audit_marshawn_geometry.py')]},
                limitation='Linearized fixed-camera independent0.5HDpixel localization assumption only; excludes calibration bias, shadow/feature-center bias and panel deformation; not a physical safety band or a plane fit.')
    with a.output.open('x')as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['review','transforms','box','output']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
