"""CPU analytic checks for actual runtime optical-depth and gradient algebra."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import importlib.util
import math
from pathlib import Path
import unittest
import numpy as np
import torch

spec=importlib.util.spec_from_file_location('geometry',Path(__file__).parents[1]/'scripts/marshawn_focused_clothing_geometry.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ConstantField:
    def __init__(self):
        self.sigma=torch.nn.Parameter(torch.tensor(5.));self.calls=0
    def get_density(self,ray_samples):
        self.calls+=1
        return self.sigma.expand(ray_samples.frustums.origins.shape[0],1),None


class ConstantModel:
    device=torch.device('cpu')
    def __init__(self):self.field=ConstantField()
    def semantic_lookup(self,xyz):return torch.ones(xyz.shape[0],dtype=torch.bool)


class IntegralTests(unittest.TestCase):
    def make(self,arm):
        ids=np.array([100+i%31 for i in range(84)])
        arrays=dict(point_idx=ids,camera_idx=np.arange(84)%3,origins=np.zeros((84,3)),directions=np.tile([0,0,1.],(84,1)),t_start=np.zeros(84),t_end=np.full(84,.95),witness_depth=np.ones(84),witness_xyz=np.tile([0,0,1.],(84,1)))
        rows=[dict(index=i,point=int(ids[i]),depth=1.,prefix_end=.95,band_start=.99,band_end=1.01)for i in range(84)]
        model=ConstantModel();return model,m.FocusedClothingGeometry(model,arrays,rows,'a'*64,arm)
    def test_constant_density_band_and_native_gradient(self):
        model,aux=self.make('closer_prefix_band');before=torch.get_rng_state().clone()
        value=aux.loss();value.backward()
        expected=.003*5*.98-.001*math.log(-math.expm1(-5*.02))
        gradient=.003*.98-.001*.02/math.expm1(5*.02)
        self.assertAlmostEqual(float(value),expected,places=7)
        self.assertAlmostEqual(float(model.field.sigma.grad),gradient,places=7)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertFalse(torch.cuda.is_initialized())
    def test_control_draws_without_density_work(self):
        model,aux=self.make('control');self.assertEqual(float(aux.loss()),0.)
        self.assertEqual(model.field.calls,0);self.assertEqual(aux.stream.draws,1)
    def test_both_prefix_intervals(self):
        for arm,length in [('old_prefix',.95),('closer_prefix',.98)]:
            model,aux=self.make(arm);value=aux.loss();value.backward()
            self.assertAlmostEqual(float(value),.003*5*length,places=7)
            self.assertAlmostEqual(float(model.field.sigma.grad),.003*length,places=7)


if __name__=='__main__':unittest.main()
