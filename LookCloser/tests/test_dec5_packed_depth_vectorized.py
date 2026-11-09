import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dec5_hair import packed_depth_quantile_vectorized as packed_depth_quantile,packed_depth_quantile_reference


def test_ragged_shuffled_cdfs_match_reference_including_exact_plateaus_and_invalid_rays():
 rng=np.random.RandomState(42);starts=[];ends=[];weights=[];ids=[]
 for ray in range(128):
  n=rng.randint(1,513);edge=np.r_[0,np.cumsum(rng.uniform(.001,1,n))];edge/=edge[-1];w=rng.dirichlet(np.ones(n))*rng.uniform(.05,.99)
  if ray%9==0:w[:]=0;w[0]=.25;w[-1]=.25
  if ray%13==0:w[n//2]=np.nan
  starts.extend(edge[:-1]);ends.extend(edge[1:]);weights.extend(w);ids.extend([ray]*n)
 order=rng.permutation(len(ids));v=[np.asarray(x)[order] for x in [starts,ends,weights,ids]];near=np.linspace(.01,1,128);far=near+3
 for q in [.1,.5,.9]:
  actual=packed_depth_quantile(*v,near,far,q);expected=packed_depth_quantile_reference(*v,near,far,q)
  np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-14);np.testing.assert_array_equal(actual.astype(np.float32),expected.astype(np.float32))


def test_global_prefix_offset_cannot_move_a_zero_weight_quantile_plateau():
 starts=np.tile([0.,.25,.75],3);ends=np.tile([.25,.75,1.],3);weights=np.array([.1,.11,.13,.25,0,.25,.2,.12,.21]);ids=np.repeat(np.arange(3),3)
 actual=packed_depth_quantile(starts,ends,weights,ids,np.zeros(3),np.ones(3));expected=packed_depth_quantile_reference(starts,ends,weights,ids,np.zeros(3),np.ones(3))
 np.testing.assert_array_equal(actual,expected)


def test_opacity_floor_reveals_low_mass_rays_without_changing_existing_depths():
 starts=np.zeros(5);ends=np.ones(5);weights=np.array([0,.005,.03,.1,.5]);ids=np.arange(5)
 near=np.ones(5);far=np.full(5,3.)
 for method in [packed_depth_quantile,packed_depth_quantile_reference]:
  default=method(starts,ends,weights,ids,near,far,quantile=.1)
  low=method(starts,ends,weights,ids,near,far,quantile=.1,minimum_mass=.001)
  np.testing.assert_array_equal(default>0,[False,False,False,True,True])
  np.testing.assert_array_equal(low>0,[False,True,True,True,True])
  np.testing.assert_array_equal(default[default>0],low[default>0])
  assert method(starts,ends,weights,ids,near,far,minimum_mass=0)[0]==0
