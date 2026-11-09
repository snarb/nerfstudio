"""Narrow tagged-Path relocation after two preserved source-parity guard failures."""
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import yaml

C=Path('/fsx/tmp/marshawn_001660_hd/tuning_20261106')
SC='/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106'
OLD=C/'frozen_LC_semantic_source_silver_parity_plan_v1'
P=C/'frozen_LC_semantic_source_silver_parity_pathfix_v2'
R=SC+'/'+P.name
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')

def main():
    P.mkdir(exist_ok=True)
    shutil.copytree(OLD/'code',P/'code',dirs_exist_ok=True)
    original=Path(C/'frozen_LC_semantic_capture_plan_v2/local_source/config_local_paths_only.yml').read_text()
    text=original.replace(str(C),SC)
    changed=[]
    for indent in ['', '    ']:
        old=''.join(indent+'- '+part+'\n' for part in C.parts)
        new=''.join(indent+'- '+part+'\n' for part in Path(SC).parts)
        count=text.count(old)
        if count:text=text.replace(old,new);changed.append(dict(indent=len(indent),nodes=count))
    if sum(x['nodes'] for x in changed)!=2:raise ValueError('Expected exactly output_dir/data tagged paths')
    (P/'config_execution_alias.yml').write_text(text)
    # Reconstruct path-valued YAML nodes without invoking object constructors.
    def normalize(n,relocate):
        if n.tag.endswith('pathlib.PosixPath'):
            value=Path(*[x.value for x in n.value]);s=str(value)
            if relocate:s=s.replace(str(C),SC)
            return [n.tag,s]
        if isinstance(n,yaml.MappingNode):return [n.tag,[[normalize(k,relocate),normalize(v,relocate)]for k,v in n.value]]
        if isinstance(n,yaml.SequenceNode):return [n.tag,[normalize(v,relocate)for v in n.value]]
        return [n.tag,n.value.replace(str(C),SC)if relocate else n.value]
    if normalize(yaml.compose(original),True)!=normalize(yaml.compose(text),False):
        raise ValueError('Config changes beyond exact string and tagged-Path relocation')
    m=json.loads((OLD/'source_execution_manifest.json').read_text())
    m.update(local_config=R+'/config_execution_alias.yml',output=R+'/parity')
    m['bound_inputs'][R+'/config_execution_alias.yml']=sha(P/'config_execution_alias.yml')
    m['bound_inputs'].update({q['archive']:q['archive_sha256']for q in m['records']})
    write(P/'source_execution_manifest.json',m)
    sl=json.loads((OLD/'source_loader_release_TEMPLATE.json').read_text());sl['manifest_sha256']=sha(P/'source_execution_manifest.json');write(P/'source_loader_release_TEMPLATE.json',sl)
    s=json.loads((OLD/'spec.json').read_text());s.update(output=R+'/parity',root_release=R+'/root_release.json',LC_source_plan=R+'/source_execution_manifest.json',LC_source_release=R+'/source_loader_release.json');write(P/'spec.json',s)
    d=json.loads((OLD/'controller_plan.json').read_text());d['hosts']['silver'].update(controller=R+'/code/marshawn_tuning_controller.py',root=R+'/queue');j=d['jobs'][0];j.update(name='source_RGB_alpha_parity_pathfix_v2',cwd=R+'/code',metrics_file=R+'/parity/complete.json');j['command'][1]=R+'/code/run_marshawn_LC_silver_source_parity.py';j['command'][j['command'].index('--spec')+1]=R+'/spec.json';write(P/'controller_plan.json',d)
    r=json.loads((OLD/'root_review_TEMPLATE.json').read_text());r.update(spec_sha256={R+'/spec.json':sha(P/'spec.json')},controller_plan_sha256=sha(P/'controller_plan.json'));r['bound_files'].update(m['bound_inputs'])
    for name,h in m['code_sha256'].items():
        if sha(P/'code'/name)!=h:raise ValueError('Any runtime code changed')
        r['bound_files'][R+'/code/'+name]=h
    for n in ['source_execution_manifest.json','spec.json','controller_plan.json']:
        r['bound_files'][R+'/'+n]=sha(P/n)
    write(P/'root_review_TEMPLATE.json',r)
    write(P/'tagged_path_CPU_proof.json',dict(passed=True,exact_tagged_path_nodes=changed,
        config_graph_equal_after_path_relocation=True,original_config_sha256=hashlib.sha256(original.encode()).hexdigest(),
        new_config_sha256=sha(P/'config_execution_alias.yml'),runtime_code_byte_identical=True,
        source_seven_states_unchanged=True,scientific_changes=[],CUDA_started=False))
    print(json.dumps(dict(package=str(P),bindings=len(r['bound_files']),CUDA_started=False)))

if __name__=='__main__':main()
