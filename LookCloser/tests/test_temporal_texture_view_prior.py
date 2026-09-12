"""The angular preference changes selection, not RGB or geometry."""
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from temporal_texture_view_prior import angle_weights


def camera(degrees):
    pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('y',degrees,degrees=True).as_matrix()
    return {'transform_matrix':pose.tolist()}


def test_angle_prior_is_calibration_only_and_prefers_close_view():
    weights,angles=angle_weights([camera(0),camera(4),camera(90)],camera(0),4)
    np.testing.assert_allclose(angles,[0,4,90],atol=1e-5)
    np.testing.assert_allclose(weights,[1,np.exp(-.5),1e-5],rtol=1e-6)


@pytest.mark.parametrize('sigma',[0,-1,np.inf,np.nan])
def test_invalid_angular_width_is_rejected(sigma):
    with pytest.raises(ValueError):angle_weights([camera(0)],camera(0),sigma)


def test_worker_partitions_cover_150_distinct_instants():
    ids=[f'{899+2*i:06d}' for i in range(150)]
    for count in range(1,5):
        groups=[x.tolist() for x in np.array_split(ids,count)]
        assert sum(groups,[])==ids
        assert len(set(sum(groups,[])))==150
