"""Flag candidate vetoes that overlap selected actor pixels; no automatic relabel."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(master,output):
    digest=sha(master);payload=json.loads(Path(master).read_text());rows=[]
    for r in payload['rows']:
        if sha(r['candidates_npz'])!=r['source_candidate_npz_sha256']:raise ValueError('NPZ changed')
        with np.load(r['candidates_npz'],allow_pickle=False)as z:
            masks=z['masks'];black=masks[r['black_actor_instance']];grey=masks[r['grey_actor_instance']]
            actors=black|grey;count=int(actors.sum());veto=masks[r['verified_other_person_ids']].any(0)if r['verified_other_person_ids']else np.zeros_like(actors)
            comparisons=[]
            for i in r['verified_other_person_ids']:
                m=masks[i];inter=int((m&actors).sum());comparisons.append(dict(candidate=i,area=int(m.sum()),actor_overlap=inter,
                    selected_actor_fraction=inter/max(1,count),candidate_actor_fraction=inter/max(1,int(m.sum()))))
            row=dict(camera=r['camera'],selected_ids=[r['black_actor_instance'],r['grey_actor_instance']],
                selected_actor_pixels=count,union_veto_overlap=int((actors&veto).sum()),comparisons=comparisons)
            rows.append(row);print(json.dumps(row),flush=True)
    Path(output).write_text(json.dumps(dict(master=str(master),master_sha256=digest,rows=rows,
        master_changed_during_audit=sha(master)!=digest,scope='Overlap flags require actualsource review; not automatic identity or veto inference'),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--master',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.master,a.output)
