#!/usr/bin/env python3
"""Fixed-midpoint source-order counterfactual on identical frozen path geometry."""
from pathlib import Path
import argparse,json,subprocess,sys
from run_dec5_hair_campaign import atomic_json,sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['root','source-depth','target-depth','fallback','baseline','cameras','output']:p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--depth-key',default='q1000');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);n=a.root;s=Path(__file__).parent
 def run(script,*args):subprocess.run([sys.executable,str(s/script),*map(str,args)],check=True)
 frames=json.load(open(a.cameras))['frames'];anchor=a.output/'source_anchor.json';atomic_json(anchor,frames[len(frames)//2]);prediction=a.output/'prediction'
 run('render_dec5_native_head_benchmark.py','--source-rgb-data',n/'crop_native','--source-cameras',n/'source_hd_cameras.json','--source-depth',a.source_depth,'--semantics',n/'semantics_hd','--target-depth',a.target_depth,'--fallback',a.fallback,'--cameras',a.cameras,'--depth-key',a.depth_key,'--source-reference-camera',anchor,'--output',prediction)
 run('review_dec5_hair_path.py','--path',a.cameras,'--baseline',a.baseline,'--candidate',prediction,'--output',a.output/'path_review','--expected-frames',len(frames))
 run('score_dec5_path_rgb.py','--path',a.cameras,'--dynamic',a.baseline,'--fixed',prediction,'--output',a.output/'rgb_continuity.json')
 atomic_json(a.output/'complete.json',dict(prediction_receipt_sha256=sha(prediction/'complete.json'),baseline_receipt_sha256=sha(a.baseline/'complete.json'),anchor_sha256=sha(anchor),geometry_unchanged=True,source_order='fixed nearest-eight to midpoint pose; depth visibility unchanged',uses_validation_rgb=False,uses_test_rgb=False,photometric_continuity='motion-compensated proxy, not novel-view GT'))
if __name__=='__main__':main()
