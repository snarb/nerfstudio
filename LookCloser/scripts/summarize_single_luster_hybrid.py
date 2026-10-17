"""Summarize canonical native single-source evaluations, separating visual status."""
import argparse
from pathlib import Path
from luster_hybrid_common import read,write
from screen_luster_hybrid import summarize

def run(root):
    results=[]
    for candidate in sorted((root/'candidates').iterdir()):
        name=candidate.name;path=root/'evaluation'/f'{name}_metrics.json'
        if not path.exists():continue
        measured=read(path)
        if measured.get('candidate')!=name or measured.get('resolution')!='native':continue
        receipts=[candidate/f'cam{c:03d}_receipt.json' for c in [7,12,95,150]]
        if not all(p.exists() for p in receipts):continue
        records=[read(p) for p in receipts]
        if not all(r.get('texture_source_count_per_pixel')==1 and r.get('lookup_shift') is False for r in records):continue
        row=summarize(root,name);review=root/'evaluation'/f'{name}_visual_review.json'
        if review.exists():row['visual_review']=read(review).get('status','pending')
        row['accepted']=row['numeric_pass'] and row['visual_review']=='accepted'
        row['goal_achieved']=row['goal_numeric_pass'] and row['accepted']
        results.append(row)
    results.sort(key=lambda x:x['lpips'])
    write(root/'evaluation/current_single_source_scoreboard.json',results)
    print({'measured_single_source_candidates':len(results),'accepted':sum(r['accepted'] for r in results),'numerical_leader':results[0] if results else None},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
