"""Six bounded inference contracts; no technical/prod artifact is modified."""
import hashlib,json,shutil
from pathlib import Path
C=Path('/fsx/tmp/marshawn_001660_hd/tuning_20261106');SC='/cache/ubuntu/lookcloser/campaigns/marshawn_tuning_20261106'
T=C/'below_floor_density_warm_technical_plan_v1';P=C/'below_floor_density_64_readonly_plan_v1';R=SC+'/'+P.name
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(d,indent=2)+'\n')
def main():
    P.mkdir(exist_ok=True);(P/'code').mkdir(exist_ok=True)
    for n in ['replay_marshawn_below_floor64_readonly.py','compare_marshawn_below_floor64_replays_cpu.py']:
        shutil.copyfile(Path(__file__).with_name(n),P/'code'/n)
    shutil.copyfile(T/'helpers/marshawn_tuning_controller.py',P/'code/marshawn_tuning_controller.py')
    source=SC+'/distortion03_warm_continuation_candidate_v1';bound=json.loads((T/'specs/control0_full64/spec.json').read_text())['frozen_source_hashes']
    inv=json.loads((P/'selected_checkpoint_inventory.json').read_text());hashes={};jobs=[]
    for arm,gpu in [('control0',5),('prior1e4',6),('prior1e3',7)]:
        for replica in ['resume64','full64']:
            case=arm+'_'+replica;item=inv[case]
            s=dict(root_release=R+'/root_release.json',checkpoint=item['checkpoint'],checkpoint_sha256=item['sha256'],config=item['config'],
                source_code=source+'/code',source_scripts=source+'/code/LookCloser/scripts',data=SC+'/data',output=R+'/renders/'+case)
            file=P/'specs'/(case+'.json');write(file,s);hashes[R+'/specs/'+file.name]=sha(file)
            bound[R+'/specs/'+file.name]=sha(file);bound[item['config']]=item['config_sha256']
            command=[SC+'/python',R+'/code/replay_marshawn_below_floor64_readonly.py','--spec',R+'/specs/'+file.name,'--execute']
            jobs.append(dict(name=case,host='silver',gpu=gpu,command=command,cwd=R+'/code',dependencies=[arm+'_resume64']if replica=='full64'else[],
                env=dict(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',LOOKCLOSER_GPU_LEASE_RECEIPT='{job_root}/receipt.json'),metrics_file=R+'/renders/'+case+'/complete.json'))
    write(P/'comparison_spec.json',dict(ready=SC+'/data/ready.json',regions=SC+'/native_semantic_grid_20000/corrected_v2/review/region_crops.json'))
    write(P/'controller_plan.json',dict(schema_version=1,poll_seconds=10,hosts=dict(silver=dict(address='ubuntu@silver-sable',python=SC+'/python',controller=R+'/code/marshawn_tuning_controller.py',root=R+'/queue',gpus=[5,6,7])),jobs=jobs))
    for f in [*(P/'code').glob('*.py'),P/'controller_plan.json',P/'comparison_spec.json',P/'selected_checkpoint_inventory.json']:
        bound[str(f).replace(str(C),SC)]=sha(f)
    regions=C/'native_semantic_grid_20000/corrected_v2/review/region_crops.json';bound[str(regions).replace(str(C),SC)]=sha(regions)
    write(P/'root_review_TEMPLATE.json',dict(reviewer='root',readonly_replay_authorized=False,optimizer_authorized=False,private_queries_authorized=False,
        spec_sha256=hashes,controller_plan_sha256=sha(P/'controller_plan.json'),bound_files=bound,checkpoint_sha256={v['checkpoint']:v['sha256']for v in inv.values()},
        optional109_and_ND_rays_included=False,scope='Six exact trained64 checkpoints; native3FHD each, fixedV1 metrics and floatRGBA, nine fixed details/differences offline.'))
    print(json.dumps(dict(package=str(P),jobs=6,bound_files=len(bound),GPU_started=False)))
if __name__=='__main__':main()
