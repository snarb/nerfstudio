"""Bounded read-only rendering beside an authorized native producer."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path


def gpu():
    values=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(',')
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
    memory={int(p.split(',')[0]):int(p.split(',')[1]) for p in processes.splitlines() if p}
    return dict(free_mib=int(values[0]),total_mib=int(values[1]),processes=memory)


def main(a):
    b=a.base;r=b.parent.parent;n=r/'native_png_radialK4_COLD_R1_DEV3_STAGED';producer=b.parent/'native_reference_continuation_20000'
    wrapper='import torch,runpy,sys,pathlib;torch.cuda.set_per_process_memory_fraction((8*1024**3)/torch.cuda.get_device_properties(0).total_memory);p=sys.argv.pop(1);sys.path.insert(0,str(pathlib.Path(p).parent));runpy.run_path(p,run_name="__main__")'
    def status():
        s=json.loads((producer/'status.json').read_text());os.kill(s['pid'],0);return s
    def run(command,label):
        before=gpu();assert before['free_mib']>=20000,before;initial=status()
        with (a.output/(label+'.log')).open('w') as stream:
            child=subprocess.Popen([sys.executable,'-c',wrapper,*command],stdout=stream,stderr=subprocess.STDOUT)
            while True:
                code=child.poll();g=gpu();s=status();row=dict(time=time.time(),label=label,child_pid=child.pid,child_returncode=code,producer=s,gpu=g)
                with (a.output/'concurrent_health.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                if g['processes'].get(child.pid,0)>10240:
                    child.terminate();child.wait();raise RuntimeError('Read-only child exceeded10GiB limit')
                if code is not None:
                    assert code==0,(label,code);break
                time.sleep(10)
        return initial
    for camera in ['DC','HC','KA']:
        cmd=[str(b/'code/render_marshawn_native_semantic_gate.py'),'--config',str(n/'config.yml'),'--checkpoint',str(n/'immutable_selected_20000/step-000020000.ckpt'),'--checkpoint-sha256','fd56e541d91abde26b776ffe03b91caf997b3c0ada0044cebf6407728c988fa2','--box',str(b/'code/marshawn_1660_1900_floor.json'),'--semantic-grid',str(a.output/'semantic_grid.npz'),'--unprotected-semantic-grid',str(b/'occlusion_v2/W64/semantic_grid.npz'),'--camera',camera,'--output',str(a.output/'renders'/camera)]
        for p in [n/'code',r/'code_native',r/'code_sfm/LookCloser/scripts',b/'code']:cmd+=['--python-path',str(p)]
        run(cmd,'render_'+camera)
        print(json.dumps(dict(camera=camera,status='complete')),flush=True)
    run([str(b/'code/score_marshawn_support_candidate.py'),'--base',str(b),'--output',str(a.output)],'score')
    (a.output/'render_review_complete.json').write_text(json.dumps(dict(time=time.time(),producer=status(),gpu=gpu(),memory_limit_mib=10240,torch_allocator_limit_gib=8,no_producer_state_modified_by_reader=True),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
