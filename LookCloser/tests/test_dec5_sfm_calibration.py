from pathlib import Path
import json
import sqlite3
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from export_dec5_sfm_calibration import align_similarity
from study_dec5_sfm_calibration import bind_database_ids


def test_similarity_recovers_rotated_scaled_rig_without_reflection():
    source=np.random.default_rng(8).normal(size=(59,3))
    rotation=Rotation.from_rotvec([.1,-.2,.03]).as_matrix()
    target=1.7*source@rotation.T+np.array([2.,-1.,3.])
    scale,r,t=align_similarity(source,target)
    np.testing.assert_allclose(scale,1.7,atol=1e-12)
    np.testing.assert_allclose(r,rotation,atol=1e-12)
    np.testing.assert_allclose(scale*source@r.T+t,target,atol=1e-12)
    assert np.linalg.det(r) > .999999


def test_registered_model_is_bound_to_actual_database_ids(tmp_path):
    model=tmp_path/'fixed_model';model.mkdir()
    rows=[];cameras=[];images=[]
    with sqlite3.connect(str(tmp_path/'database.db')) as db:
        db.execute('CREATE TABLE images(image_id INTEGER,camera_id INTEGER,name TEXT)')
        for i in range(59):
            old=2*i+1;actual=59-i;camera=10000+actual;name='images/cam%d.png'%i
            db.execute('INSERT INTO images VALUES(?,?,?)',(actual,camera,name))
            rows.append(dict(image_id=old,name=name))
            cameras.append('%d PINHOLE 100 80 90 91 50 40'%old)
            images.extend(['%d 1 0 0 0 0 0 0 %d %s'%(old,old,name),''])
        db.commit()
    original='\n'.join(images)+'\n'
    (model/'images.txt').write_text(original)
    (model/'cameras.txt').write_text('\n'.join(cameras)+'\n')
    (model/'export_manifest.json').write_text(json.dumps(dict(images=rows)))
    result=bind_database_ids(tmp_path)
    bound=[s.split() for s in (result/'images.txt').read_text().splitlines() if s.strip()]
    assert len(bound)==59
    for i,fields in enumerate(bound):
        assert (int(fields[0]),int(fields[8]),fields[9])==(59-i,10059-i,'images/cam%d.png'%i)
    assert (model/'images.txt').read_text()==original
