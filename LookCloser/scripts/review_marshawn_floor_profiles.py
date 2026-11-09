"""Plot actual floor ray profiles and summarize convergence; no model loading."""
from pathlib import Path
import argparse,json
import numpy as np


def main(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reports={k:json.loads((root/k/'audit.json').read_text())for k in ['PNG','PQ']}
    result=dict(models={},interpretation='Local approximate-floor diagnostic only; no geometry edits or training')
    colors=dict(native='#1864ab',all_cells='#e67700',uniform8192='#868e96',uniform16384='#2b8a3e')
    for kind,report in reports.items():
        data=np.load(root/kind/'traces.npz');rows=[]
        for row in report['rows']:
            q=row['query'];profiles=row['profiles'];native=profiles['native'];allcells=profiles['all_cells'];u=profiles['uniform16384'];v=profiles['uniform8192']
            def diff(a,b,key):return float(np.max(abs(np.array(a[key])-np.array(b[key]))))
            record=dict(id=q['id'],camera=q['camera'],split=q['split'],pixel=[q['x'],q['y']],inside_reference_hull=q['inside_15point_XY_hull'],
                native_before=native['before_plane_band_mass'],native_band=native['plane_band_mass'],native_after=native['after_plane_band_mass'],
                uniform16384_before=u['before_plane_band_mass'],uniform16384_band=u['plane_band_mass'],uniform16384_after=u['after_plane_band_mass'],
                native_allcells_RGB_max_difference=diff(native,allcells,'actual_rgb'),
                native_uniform_RGB_max_difference=diff(native,u,'actual_rgb'),
                uniform_N_2N_RGB_max_difference=diff(u,v,'actual_rgb'),
                uniform_N_2N_mass_max_difference=max(diff(u,v,key)for key in ['before_plane_band_mass','plane_band_mass','after_plane_band_mass']),
                native_signed_height_quantiles=native['termination_quantiles'])
            rows.append(record)
        result['models'][kind]=dict(rows=rows,native_allcells_RGB_max_difference=max(r['native_allcells_RGB_max_difference']for r in rows),
             uniform_convergence_RGB_max=max(r['uniform_N_2N_RGB_max_difference']for r in rows),
             uniform_convergence_mass_max=max(r['uniform_N_2N_mass_max_difference']for r in rows),
             actual_saved_RGB_error_max=max(row.get('saved_RGB_max_error',0)for row in report['rows']))
        for start in range(0,len(report['rows']),3):
            fig,axes=plt.subplots(3,2,figsize=(12,9),layout='constrained')
            for j,row in enumerate(report['rows'][start:start+3]):
                q=row['query'];origin=np.array(row['origin']);direction=np.array(row['direction']);selection=json.loads((root/'selection.json').read_text());n=np.array(selection['normal']);d=selection['offset']
                for label in colors:
                    prefix=f'q{q["id"]:02d}_{label}_';starts=data[prefix+'starts'];ends=data[prefix+'ends'];sigma=data[prefix+'sigma'];weights=data[prefix+'weights'];mid=(starts+ends)/2;h=(origin+mid[:,None]*direction)@n+d
                    style='--'if label=='all_cells'else'-';width=1.8 if label in ['native','uniform16384']else.9
                    hs=(origin+starts[:,None]*direction)@n+d;he=(origin+ends[:,None]*direction)@n+d
                    # Never draw an invented density across skipped grid intervals.
                    xx=np.stack([hs,he,np.full(len(hs),np.nan)],1).reshape(-1)
                    yy=np.stack([sigma,sigma,np.full(len(hs),np.nan)],1).reshape(-1)
                    axes[j,0].plot(xx,yy,color=colors[label],ls=style,lw=width,label=label)
                    cumulative=np.cumsum(weights);previous=np.r_[0,cumulative[:-1]]
                    axes[j,1].plot(np.stack([hs,he],1).reshape(-1),np.stack([previous,cumulative],1).reshape(-1),color=colors[label],ls=style,lw=width,label=label)
                for ax in axes[j]:
                    ax.axvspan(-.03,.03,color='#74c0fc',alpha=.25);ax.axvline(0,color='black',lw=.5);ax.set_xlim(.3,-.5);ax.grid(alpha=.2);ax.set_xlabel('Signed height relative to APPROX floor plane (prepared world)')
                axes[j,0].set_yscale('log');axes[j,0].set_ylim(.01,10000);axes[j,0].set_ylabel('Actual field sigma')
                axes[j,1].set_ylabel('Cumulative termination mass');axes[j,1].set_ylim(-.02,1.02)
                prefix=f'q{q["id"]:02d}_uniform16384_'
                heights=(origin+data[prefix+'starts'][:,None]*direction)@n+d
                axes[j,1].set_xlim(float(heights.max())+.01,float(heights.min())-.01)
                axes[j,0].set_title(f'{kind} q{q["id"]:02d} {q["camera"]} {q["x"]},{q["y"]} hull={q["inside_15point_XY_hull"]}')
                axes[j,1].set_title('Full ray: air (+) → floor band → below-reference (−)')
                if j==0:axes[j,1].legend(loc='lower right',fontsize=8)
            fig.suptitle('Read-only density/termination; plane uncertainty ±0.03; NOT a carving rule')
            fig.savefig(root/f'{kind}_profiles_page{start//3+1}.png',dpi=130);plt.close(fig)
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{x:v[x]for x in v if x!='rows'}for k,v in result['models'].items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
