"""Reject source-mask crew contamination only with multiview 3D witness evidence."""
import argparse
import fcntl
import os
from pathlib import Path
import numpy as np
from common import atomic, read, sha


def qualify(data,seed_path):
    root=Path(data);ready=read(root/'native_ready.json');box=read(root/'box.json')
    lock=(root/'qualification.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
    points=[]
    for line in Path(seed_path).read_text().splitlines():
        t=line.split()
        if not t or line.startswith('#') or float(t[7])>=2 or len(t[8:])//2<3:continue
        points.append(list(map(float,t[1:4])))
    T=np.asarray(box['world_to_box']);p=np.asarray(points)@T[:3,:3].T+T[:3,3];b=np.asarray(box['aabb']);inside=((p>=b[0])&(p<=b[1])).all(1)
    for row in ready['records']:
        c=row['camera'];path=root/'parts'/(c+'.json')
        if not path.exists():continue
        info=read(path)
        if info.get('qualified'):continue
        archive=root/'parts'/(c+'.npz')
        with np.load(archive) as z:a={k:z[k] for k in z.files}
        f=row['camera_row'];pose=np.asarray(f['transform_matrix']);q=(p-pose[:3,3])@pose[:3,:3];z=-q[:,2]
        with np.errstate(divide='ignore',invalid='ignore'):
            u=np.rint(f['fl_x']*q[:,0]/z+f['cx']-.5).astype(int);v=np.rint(-f['fl_y']*q[:,1]/z+f['cy']-.5).astype(int)
        valid=(z>0)&(u>=0)&(v>=0)&(u<f['w'])&(v<f['h']);idx=np.flatnonzero(valid);lab=a['labels'][v[idx],u[idx]]
        crew=np.zeros(a['labels'].shape,bool);retained=[];rejected=[]
        for r in info['records']:
            subset=lab==r['id'];n=int(subset.sum());n_inside=int(inside[idx][subset].sum());r=dict(r,seed_witnesses=n,inside_aabb_witnesses=n_inside)
            if n>=8 and n_inside/n<.2:
                crew|=a['labels']==r['id'];rejected.append(r)
            else:retained.append(r)
        if not retained:raise ValueError('No performer retained '+c)
        a['crew_exclusion']=crew;a['unknown_people']|=crew
        for r in rejected:
            a['labels'][crew]=0
            a.pop('person'+str(r['id']),None);a.pop('head'+str(r['id']),None)
        a['head_union']&=~crew
        tmp=archive.with_name(archive.stem+'.qualification.%s.npz'%os.getpid())
        np.savez_compressed(tmp,**a);os.replace(tmp,archive)
        atomic(path,dict(info,records=retained,rejected_crew=rejected,qualified=True,qualification='COLMAP error<2, tracks>=3; instance with>=8 projected witnesses and<20% inside accepted common AABB excluded; low evidence remains UNKNOWN/review',seed_source_sha256=sha(seed_path),archive_sha256=sha(archive)))
        print('QUALIFY',c,'retained',len(retained),'crew',[r['id'] for r in rejected],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--seeds',required=True);a=p.parse_args();qualify(a.data,a.seeds)
