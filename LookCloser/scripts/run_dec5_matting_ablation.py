#!/usr/bin/env python3
"""One frozen-geometry native source-alpha/compositing ablation and RGB gate."""
from pathlib import Path
import argparse,json,subprocess,sys
from run_dec5_hair_campaign import atomic_json,sha


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--style',choices=['binary','alpha','fb'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);s=Path(__file__).parent;n=a.root;root=a.campaign
 def run(script,*args):subprocess.run([sys.executable,str(s/script),*map(str,args)],check=True)
 prediction=a.output/'prediction';metrics=a.output/'metrics.json'
 run('render_dec5_native_matted.py','--data',n/'crop_native','--source-cameras',n/'source_hd_cameras.json','--source-depth',root/'artifacts/geometry_v2_fg5k/sources','--semantics',n/'semantics_hd','--matting',root/'artifacts/native_mattes_srgb_f32','--parent',root/'artifacts/geometry_v2_fg5k/q1000_prediction','--fallback',n/'ray_bg_fallback_uhd_hull_v2','--cameras',n/'validation_4k_cameras.json','--output',prediction,'--style',a.style)
 run('score_dec5_hair.py','--root',n/'benchmark_scene','--prediction',prediction,'--output',metrics,'--boundary-radius','5')
 atomic_json(a.output/'complete.json',dict(style=a.style,metrics_sha256=sha(metrics),prediction_receipt_sha256=sha(prediction/'complete.json'),geometry_unchanged=True,uses_validation_rgb=False,uses_test_rgb=False))
if __name__=='__main__':main()
