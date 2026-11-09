#!/usr/bin/env python3
"""Evaluate frozen camera candidates once all independent frame detections exist."""
from marshawn_printed_target_refine_independent import *
import hashlib
summary={}
for stem,file in [('square','candidate_cameras_DO_NOT_APPLY.npz'),('freecorners','freecorners_candidate_DO_NOT_APPLY.npz')]:
 a=np.load(OUT/file);r=a['R'];t=a['t'];summary[stem]={}
 for f in [408,448,488]:
  o=read(f);summary[stem][f]=dict(camera_marker_detections=len(o),unique_cameras=len(set(c for c,m in o)),unique_ids=sorted(set(m for c,m in o)),metrics={})
  for tag,rr,tt in [('baseline',R0,t0),('candidate',r,t)]:summary[stem][f]['metrics'][tag]=dict(epipolar_px=epipolar(o,rr,tt),crossview_reprojection_px=cross_reproject(o,rr,tt))
# Replace provisional metrics emitted while extraction could still be active.
for stem,filename in [('square','refinement_summary.json'),('freecorners','freecorners_summary.json')]:
 p=OUT/filename;v=json.loads(p.read_text());v['metrics']=summary[stem];v['metrics_schema']='Final per-frame detection coverage + metrics/baseline,candidate; final_comparison.json';p.write_text(json.dumps(v,indent=2))
# Temporal difference of independent corner extractions; not treated as calibration GT.
summary['temporal_corner_displacements_px']={}
o=read(408)
for f in [448,488]:
 z=read(f);common=set(o)&set(z);e=np.concatenate([np.linalg.norm(o[k]-z[k],axis=1) for k in common]);summary['temporal_corner_displacements_px'][f]=stat(e)
summary['rig_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in __import__('marshawn_printed_target_independent').RIG.glob('*.txt') if p.name in ['cameras.txt','images.txt']}
summary['script_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('marshawn_printed_target*independent.py')}
(OUT/'final_comparison.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
