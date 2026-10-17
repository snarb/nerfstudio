"""Propose every TRAIN camera with local geometric support, without global vote truncation."""
import argparse
from pathlib import Path
import numpy as np
from luster_hybrid_common import allowed_sources,geometry,native_rows,read,sha,unproject,write
from transfer_luster_hybrid import source_score

def prepare(root,step=8):
    out=root/'variants/expanded_sources';out.mkdir(parents=True,exist_ok=True)
    for key in ['geometry','baseline','evaluation','candidates','logs','runtime','code','warps']:
        path=out/key
        if not path.exists():path.symlink_to(root/key,target_is_directory=True)
    (out/'preflight').mkdir(exist_ok=True)
    for p in (root/'preflight').iterdir():
        q=out/'preflight'/p.name
        if p.name!='sources.json' and not q.exists():q.symlink_to(p,target_is_directory=p.is_dir())
    rows=native_rows();sources={};statistics={}
    for c in [7,12,95,150]:
        row=rows[c];g=geometry(root,c);yy,xx=np.mgrid[0:row['h']:step,0:row['w']:step]
        good=np.isfinite(g['depth_z_q50'][yy,xx])&(g['alpha'][yy,xx]>=.99)
        xy=np.stack([xx,yy],-1)[good].astype(np.float32);world=unproject(g['depth_z_q50'][yy,xx][good],row,xy)
        candidates=[]
        for source in allowed_sources(rows,c):
            score,ratio=source_score(world,row,rows[source]);valid=(score<=1)&(ratio>=.5);count=int(valid.sum())
            if count:candidates.append(dict(camera=source,grid_points=count,coverage=count/len(world)))
        candidates.sort(key=lambda x:(-x['grid_points'],x['camera']));sources[str(c)]=[x['camera'] for x in candidates];statistics[str(c)]=candidates
    all_sources=sorted(set(s for ss in sources.values() for s in ss))
    result=dict(sources=sources,all_sources=all_sources,count=max(map(len,sources.values())),selection='all allowed cameras with any predicted surface point passing angular/resolution/in-frame gates; no top-k vote truncation',grid_step=step,statistics=statistics,uses_target_rgb=False,uses_masks=False,code_sha256=sha(Path(__file__)))
    destination=out/'preflight/sources.json'
    if destination.exists():assert read(destination)==result
    else:write(destination,result)
    missing=[s for s in all_sources if not (root/f'geometry/cam{s:03d}/receipt.json').exists()]
    write(root/'preflight/expanded_sources_request.json',dict(variant=str(out),missing_geometry=missing,source_lists_sha256=sha(destination)))
    print(dict(cameras={c:len(ss) for c,ss in sources.items()},missing_geometry=missing),flush=True)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();prepare(a.root)
