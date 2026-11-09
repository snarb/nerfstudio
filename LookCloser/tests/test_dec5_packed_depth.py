from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dec5_hair import packed_depth_quantile


def test_native_median_is_exact_for_constant_density_under_nonuniform_subdivision():
    expected=-np.log((1+np.exp(-1))/2)
    for edges in [np.array([0.,1.]),np.array([0.,.1,.3,1.]),np.linspace(0,1,101)]:
        weights=np.exp(-edges[:-1])-np.exp(-edges[1:])
        result=packed_depth_quantile(edges[:-1],edges[1:],weights,np.zeros(len(weights),int),[2.],[5.])
        np.testing.assert_allclose(result,[2+3*expected],rtol=1e-10)


def test_packed_quantile_keeps_rays_separate_and_empty_rays_invalid():
    edges=np.array([0.,.2,1.]);w=np.exp(-edges[:-1])-np.exp(-edges[1:])
    value=packed_depth_quantile([0.,0.,.2],[.2,1.,1.],[w[0],1-np.exp(-1),w[1]],[0,2,0],[0,0,2],[1,1,5])
    expected=-np.log((1+np.exp(-1))/2)
    np.testing.assert_allclose(value,[expected,0,2+3*expected],rtol=1e-10)


def test_spatial_layer_quantile_keeps_attenuation_from_another_front_layer():
    from dec5_hair import packed_layer_depth_quantile
    edges=np.arange(4,dtype=np.float64);weights=np.exp(-edges[:-1])*(-np.expm1(-1.))
    result,mass=packed_layer_depth_quantile(edges[:-1]/3,edges[1:]/3,weights,np.zeros(3,int),[0.],[3.],[False,True,False])
    expected=1-np.log((1+np.exp(-1.))/2)
    np.testing.assert_allclose(result,[expected],rtol=0,atol=1e-12)
    np.testing.assert_allclose(mass,[np.exp(-1.)*(1-np.exp(-1.))],rtol=0,atol=1e-12)


def test_low_opacity_layer_partition_invariance_and_empty_rays():
    from dec5_hair import packed_layer_depth_quantile
    results=[]
    for count in [3,12]:
        edges=np.linspace(0,3,count+1);sigma=.01;weights=np.exp(-sigma*edges[:-1])*(-np.expm1(-sigma*np.diff(edges)))
        mask=(edges[:-1]>=1)&(edges[1:]<=2)
        depth,mass=packed_layer_depth_quantile(edges[:-1]/3,edges[1:]/3,weights,np.zeros(count,int),[0.,0.],[3.,3.],mask,minimum_mass=.001)
        assert depth[1]==0 and mass[1]==0 and 0<mass[0]<.01
        results.append(depth[0])
    np.testing.assert_allclose(results,1-np.log((1+np.exp(-.01))/2)/.01,rtol=0,atol=1e-11)


def test_front_quantiles_match_analytic_nonuniform_volume():
    edges=np.array([0.,.03,.2,.75,1.]);weights=np.exp(-edges[:-1])-np.exp(-edges[1:])
    for q in [.1,.25,.75]:
        value=packed_depth_quantile(edges[:-1],edges[1:],weights,np.zeros(len(weights),int),[2.],[5.],quantile=q)
        expected=2-3*np.log1p(-q*(-np.expm1(-1.)))
        np.testing.assert_allclose(value,[expected],rtol=0,atol=1e-12)
