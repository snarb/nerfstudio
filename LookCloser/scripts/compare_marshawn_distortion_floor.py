"""Same sealed floor rays: compare field concentration before/after distortion."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def spread(data,prefix,origin,direction,n,d):
    starts,ends,weights=[data[prefix+k].astype(float)for k in ['starts','ends','weights']]
    mid=(starts+ends)/2;mass=weights.sum();p=weights/mass;h=(origin+mid[:,None]*direction)@n+d
    def quantile(frac):return h[min(int(np.searchsorted(np.cumsum(p),frac)),len(h)-1)]
    mean=(p*h).sum();q10,q50,q90=[quantile(q)for q in [.1,.5,.9]]
    order=np.argsort(h);hs=h[order];ps=p[order];priorp=np.r_[0,np.cumsum(ps)[:-1]];priorh=np.r_[0,np.cumsum(ps*hs)[:-1]]
    return dict(opacity=float(mass),mean_signed_height=float(mean),std_height=float(np.sqrt((p*(h-mean)**2).sum())),
                height_q10=float(q10),height_q50=float(q50),height_q90=float(q90),height_central80_width=float(abs(q90-q10)),
                pairwise_mean_absolute_height=float(2*np.sum(ps*(hs*priorp-priorh))))


def main(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    old=json.loads((root/'PNG/audit.json').read_text());new=json.loads((root/'DISTORTION/audit.json').read_text())
    if old['selection_sha256']!=new['selection_sha256']:raise ValueError('Selections changed')
    selection=json.loads((root/'selection.json').read_text());n=np.array(selection['normal']);d=selection['offset']
    datasets={name:np.load(root/name/'traces.npz')for name in ['PNG','DISTORTION']};rows=[]
    for before,after in zip(old['rows'],new['rows']):
        if before['query']!=after['query']:raise ValueError('Query changed')
        q=before['query'];o=np.array(before['origin']);v=np.array(before['direction'])
        if not np.array_equal(before['origin'],after['origin'])or not np.array_equal(before['direction'],after['direction']):raise ValueError('Actualrays differ')
        record=dict(query=q,profiles={})
        for label in ['native','all_cells','uniform8192','uniform16384']:
            a=spread(datasets['PNG'],f'q{q["id"]:02d}_{label}_',o,v,n,d);b=spread(datasets['DISTORTION'],f'q{q["id"]:02d}_{label}_',o,v,n,d)
            record['profiles'][label]=dict(old=a,new=b,central80_width_ratio=b['height_central80_width']/a['height_central80_width'],
                old_band_mass=before['profiles'][label]['plane_band_mass'],new_band_mass=after['profiles'][label]['plane_band_mass'],
                old_after_mass=before['profiles'][label]['after_plane_band_mass'],new_after_mass=after['profiles'][label]['after_plane_band_mass'])
        native=after['profiles']['native'];allcells=after['profiles']['all_cells'];u=after['profiles']['uniform16384'];v8=after['profiles']['uniform8192']
        record['checks']=dict(native_allcells_RGB=float(np.max(abs(np.array(native['actual_rgb'])-allcells['actual_rgb']))),
            uniform_N_2N_RGB=float(np.max(abs(np.array(u['actual_rgb'])-v8['actual_rgb']))),
            uniform_N_2N_mass=max(abs(u[k]-v8[k])for k in ['before_plane_band_mass','plane_band_mass','after_plane_band_mass']))
        rows.append(record)
    for start in range(0,len(rows),3):
        fig,axes=plt.subplots(3,2,figsize=(12,9),layout='constrained')
        for j,row in enumerate(rows[start:start+3]):
            q=row['query'];o=np.array(old['rows'][start+j]['origin']);v=np.array(old['rows'][start+j]['direction'])
            limits=[]
            for kind,label,color in [('PNG','old8k uniform16384','#c92a2a'),('DISTORTION','new8k uniform16384','#1864ab')]:
                z=datasets[kind];prefix=f'q{q["id"]:02d}_uniform16384_';st,en,w,sigma=[z[prefix+k]for k in ['starts','ends','weights','sigma']]
                hs=(o+st[:,None]*v)@n+d;he=(o+en[:,None]*v)@n+d;xx=np.stack([hs,he,np.full(len(hs),np.nan)],1).reshape(-1);yy=np.stack([sigma,sigma,np.full(len(hs),np.nan)],1).reshape(-1)
                axes[j,0].plot(xx,yy,color=color,lw=1.1,label=label);cw=np.cumsum(w)
                axes[j,1].plot(np.stack([hs,he],1).reshape(-1),np.stack([np.r_[0,cw[:-1]],cw],1).reshape(-1),color=color,lw=1.5,label=label);limits.extend([hs.max(),he.min()])
            for ax in axes[j]:ax.axvspan(-.03,.03,color='#74c0fc',alpha=.25);ax.axvline(0,c='black',lw=.5);ax.grid(alpha=.2);ax.set_xlabel('Signed height versus APPROX floor (prepared world)')
            axes[j,0].set_xlim(.2,-.35);axes[j,0].set_yscale('log');axes[j,0].set_ylim(.01,1e5);axes[j,0].set_ylabel('Actual field sigma');axes[j,0].set_title(f'q{q["id"]:02d} {q["camera"]} {q["x"]},{q["y"]} hull={q["inside_15point_XY_hull"]}')
            axes[j,1].set_xlim(max(limits)+.01,min(limits)-.01);axes[j,1].set_ylim(-.02,1.02);axes[j,1].set_ylabel('Full-ray cumulative termination mass');axes[j,1].legend(fontsize=8,loc='lower right')
        fig.suptitle('Same9 rays; grid-bypassing uniform16384 field comparison; plane band NOT depth truth')
        fig.savefig(root/'DISTORTION'/f'old_new_profiles_page{start//3+1}.png',dpi=130);plt.close(fig)
    ratios=np.array([row['profiles']['uniform16384']['central80_width_ratio']for row in rows]);result=dict(rows=rows,
        median_uniform_central80_width_ratio=float(np.median(ratios)),max_uniform_N_2N_mass=max(row['checks']['uniform_N_2N_mass']for row in rows),
        max_uniform_N_2N_RGB=max(row['checks']['uniform_N_2N_RGB']for row in rows),max_native_allcells_RGB=max(row['checks']['native_allcells_RGB']for row in rows),
        source_sha256={k:sha(root/k/'audit.json')for k in ['PNG','DISTORTION']},selection_sha256=old['selection_sha256'],
        limitation='Old native used acceptedguard; newnative isactualcore0. Uniform16384 is the primary field-only comparison. Planeoffset affects signed depth but not width/concentration ratio.',
        default_point_membership=new['default_point_membership'],new_original_state_restored=new['state_unchanged'])
    (root/'DISTORTION/comparison_old8k.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k not in ['rows','default_point_membership']}))
    for row in rows:
        p=row['profiles']['uniform16384'];print(row['query']['id'],row['query']['camera'],p['old']['height_central80_width'],p['new']['height_central80_width'],p['central80_width_ratio'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
