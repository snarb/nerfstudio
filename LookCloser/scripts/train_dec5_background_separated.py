#!/usr/bin/env python3
"""Explicit warm-start experiment: frozen background + full LookCloser foreground.

Reuses the established trainer, with fresh optimizer at 0.1 base LR, mature
subject quotas and no repeated traversal warmup. Background is fitted separately
on train59 BG only. Its weights are an immutable external checkpoint dependency.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,sys,torch
from run_dec5_hair_campaign import atomic_json,sha


def main():
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--background',type=Path,required=True);p.add_argument('--background-semantics',type=Path,required=True);p.add_argument('--empty-space-weight',type=float,default=.05);p.add_argument('--known-background-rgb',action='store_true');a,rest=p.parse_known_args();output=Path(rest[rest.index('--output')+1])
 if '--initial-checkpoint' not in rest or '--resume-from' in rest or '--full-lookcloser' not in rest:raise ValueError('Explicit full-LookCloser weight-only warmstart required')
 import train_dec5_hair_baseline as baseline
 import dec5_lookcloser_training as helpers
 import dec5_subject_sampling as sampling
 from dec5_background_separation import install_training
 from nerfstudio.pipelines.lookcloser_pipeline import LookCloserPipeline
 from export_dec5_hair_volume import fingerprint
 original_configure=helpers.configure
 def configure(cfg,*args,**kwargs):
  original_configure(cfg,*args,**kwargs)
  for key in ['adaptive_warmup_steps','occupancy_warmup_steps','occupancy_binary_warmup_steps']:setattr(cfg.pipeline.model,key,0)
 helpers.configure=configure
 original_sampler=sampling.SubjectSampler
 class MatureSampler(original_sampler):
  def __init__(self,*args,**kwargs):kwargs.update(warmup=0,ramp=1);super().__init__(*args,**kwargs)
 sampling.SubjectSampler=MatureSampler
 original_load=LookCloserPipeline.load_state_dict;installed=[]
 def load(pipe,*args,**kwargs):
  result=original_load(pipe,*args,**kwargs)
  if installed:raise ValueError('Unexpected repeated pipeline load during background experiment')
  before=fingerprint(pipe.model);provider,counts=install_training(pipe,a.background,a.background_semantics,output,a.empty_space_weight,a.known_background_rgb)
  if fingerprint(pipe.model)!=before:raise ValueError('Background attachment changed foreground parameters')
  request=json.load(open(output/'request.json'));request['ray_background']=dict(path=str(a.background),sha256=sha(a.background),empty_space_weight=a.empty_space_weight,known_background_rgb=a.known_background_rgb);request['background_separation_experiment']=True;request['warmstart_policy']='weights only, fresh 0.1 base LR optimizer, mature subject quotas, traversal warmups disabled';atomic_json(output/'request.json',request)
  pipe.model.current_train_step=0;initial=dict(step=0,**baseline.evaluate(SimpleNamespace(pipeline=pipe),output/'eval_000000',stride=request['evaluation_stride']))
  torch.save(dict(step=0,pipeline=pipe.state_dict()),output/'gate_000000.pt');atomic_json(output/'background_initial_rgb.json',initial);installed.append((provider,counts));return result
 LookCloserPipeline.load_state_dict=load;baseline._background_warmstart_installer_active=True;sys.argv=[sys.argv[0]]+rest+['--background-warmstart-experiment'];baseline.main()
 if len(installed)!=1:raise RuntimeError('Background was not attached')
 initial=json.load(open(output/'background_initial_rgb.json'));history=[initial]+json.load(open(output/'history.json'));atomic_json(output/'history.json',history);best=baseline.publish_best(history,output);atomic_json(output/'selection.json',best);progress=json.load(open(output/'progress.json'));progress['selection']=best;atomic_json(output/'progress.json',progress);installed[0][0].verify();atomic_json(output/'background_training_audit.json',dict(frozen_background_unchanged=True,**installed[0][1],initial_foreground_fingerprint_unchanged=True,initial_rgb_function_rebased=True))
if __name__=='__main__':main()
