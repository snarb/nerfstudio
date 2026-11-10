import os,sys,json,hashlib,subprocess
from pathlib import Path
r=Path('/cache/ubuntu/lookcloser/campaigns/marshawn908_20261109');o=r/'diagnostics/bat923';sys.path[:0]=[str(o),str(r/'tools'),str(r/'frozen_helpers'),str(r/'runtime_base/LookCloser/scripts')]
from admission_vote_variant import build
src=r/'frames/000923/data_model';data=o/'data_votes3';out=o/'votes3_trial';out.mkdir(exist_ok=True)
if not (data/'admission/manifest.json').exists():
 temp=o/f'data_votes3_build_{os.getpid()}';m=build(src,temp,3);temp.rename(data)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if not (out/'spec.json').exists():
 s=json.loads((r/'frames/000923/train/spec.json').read_text());s['data']=str(data);s['output']=str(out);s['stage_plan']=str(out/'stage.json');s['semantic_admission']=dict(manifest=str(data/'admission/manifest.json'),manifest_sha256=sha(data/'admission/manifest.json'));s['head_proxy_manifest']=str(data/'heads/manifest.json');s['head_proxy_manifest_sha256']=sha(data/'heads/manifest.json');s['code_sha256'][str(o/'admission_vote_variant.py')]=sha(o/'admission_vote_variant.py');s['code_sha256'][str(data/'admission/manifest.json')]=sha(data/'admission/manifest.json');(out/'stage.json').write_text(json.dumps(dict(stop_after_iterations=16000)));(out/'spec.json').write_text(json.dumps(s,indent=2))
subprocess.run(['/home/ubuntu/lookcloser/envs/marshawn_leader/bin/python',str(r/'tools/worker_background.py'),'--spec',str(out/'spec.json')],check=True)
