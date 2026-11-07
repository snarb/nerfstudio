"""Source-bound wrapper for the read-only all66 nominal-contour audit.

This supplies a necessary publication check, not a physical occupancy claim.
No model is loaded and no DATA or grid file is written.
"""
from pathlib import Path
import argparse, importlib.util, json, sys, hashlib
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qualify(data,admission,output,audit_code,ready_sha,manifest_sha):
    if sha(data/'ready.json')!=ready_sha or sha(admission/'manifest.json')!=manifest_sha:
        raise ValueError('Current source/grid differs from requested release')
    source=data.parents[2]/'code';sys.path[:0]=[str(source),str(source/'LookCloser/scripts')]
    from marshawn_video_support import load_support
    from marshawn_video_metrics import actor_visibility
    ready=json.loads((data/'ready.json').read_text());tf=json.loads((data/'transforms.json').read_text())
    cameras={r['physical_camera']for r in tf['frames']if r['file_path']in tf['train_filenames']}
    if ready['train']!=66 or len(cameras)!=66 or cameras!={r['camera']for r in ready['records']}:
        raise ValueError('Exactly all66 current training cameras required')
    mapped,proof=load_support(admission/'manifest.json',manifest_sha,data,np.asarray(ready['aabb']),(128,)*3)
    visibility=[]
    for row in ready['records']:
        with np.load(data/'semantics'/(row['camera']+'.npz'),allow_pickle=False)as z:
            for actor in ['black','grey']:
                state=actor_visibility(row,actor,z[actor])
                if state=='present'and not z[actor].any():raise ValueError('Visible actor has no positive contour')
                visibility.append(dict(camera=row['camera'],actor=actor,status=state))
    spec=importlib.util.spec_from_file_location('_qualified_read_only_contour',audit_code)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    report=module.audit(data,admission,output)
    records=report['records'];counts=report['counts_by_grid']
    if len(records)!=66*2*3 or {r['camera']for r in records}!=cameras:
        raise ValueError('Dense audit omitted current camera/actor/grid pairs')
    passed=all(counts[k]['failed']==0 for k in ['source_pad1','source_mapped'])
    if sha(data/'ready.json')!=ready_sha or sha(admission/'manifest.json')!=manifest_sha:
        raise ValueError('Source/grid changed during dense qualification')
    result=dict(kind='ALL66_DENSE_NOMINAL_CONTOUR_SOURCE_QUALIFICATION_V1',passed=passed,frame=ready['frame'],
                all66=True,ready_sha256=ready_sha,admission_manifest_sha256=manifest_sha,frozen_load_support=proof,
                report=str(output/'report.json'),report_sha256=sha(output/'report.json'),counts_by_grid=counts,
                audit_code_sha256=sha(audit_code),qualifier_code_sha256=sha(__file__),visibility=visibility,
                source_mapped_bool_sha256=hashlib.sha256(mapped.tobytes()).hexdigest(),
                no_inputs_model_grid_or_dispatch_mutated=True,
                limitations=['Nominal foreground contour retention is necessary, not a physical hidden-depth certificate.',
                             'Missing native hair absent from nominal masks requires the separate source-qualified native audit.',
                             'This source check does not certify the later learned occupancy grid.'])
    (output/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(frame=ready['frame'],passed=passed,counts_by_grid=counts,qualification=str(output/'qualification.json'))),flush=True)
    return passed


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','admission','output','audit-code']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--ready-sha',required=True);p.add_argument('--manifest-sha',required=True)
    a=p.parse_args();sys.exit(0 if qualify(a.data,a.admission,a.output,a.audit_code,a.ready_sha,a.manifest_sha) else 2)
