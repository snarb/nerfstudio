"""Independent geometric oracle for the read-only contour DDA diagnostic."""
import json
import numpy as np
from marshawn_admission_contour_audit import ray_hits


def run():
    rng=np.random.default_rng(13);grid=rng.random((8,8,8))>.9
    bounds=np.array([[-1.,-1.,-1.],[1.,1.,1.]])
    origins=rng.normal(size=(1000,3))*3;directions=rng.normal(size=(1000,3))
    got,_=ray_hits(grid,bounds,origins,directions)
    cells=np.argwhere(grid);lower=bounds[0]+cells/8*(bounds[1]-bounds[0]);upper=lower+(bounds[1]-bounds[0])/8
    expected=[]
    for origin,direction in zip(origins,directions):
        a=(lower-origin)/direction;b=(upper-origin)/direction
        entry=np.maximum(np.minimum(a,b).max(1),0.)
        exit=np.maximum(a,b).min(1)
        expected.append(bool((exit>entry).any()))
    assert np.array_equal(got,expected)
    for axis in range(3):
        direction=np.zeros(3);direction[axis]=1
        origin=np.zeros(3);origin[axis]=-2
        hit,valid=ray_hits(np.ones((8,8,8),bool),bounds,origin,np.array([direction]))
        assert hit[0]and valid[0]
        hit,valid=ray_hits(np.zeros((8,8,8),bool),bounds,origin,np.array([direction]))
        assert not hit[0]and valid[0]
        origin[(axis+1)%3]=2
        hit,valid=ray_hits(np.ones((8,8,8),bool),bounds,origin,np.array([direction]))
        assert not hit[0]and not valid[0]
    print(json.dumps(dict(passed=True,random_rays_vs_exhaustive_occupied_cell_slab_oracle=1000,
                          axis_parallel_full_empty_and_missed_grids=9)))


if __name__=='__main__':run()
