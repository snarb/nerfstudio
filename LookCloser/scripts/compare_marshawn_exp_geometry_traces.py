"""Matched103-ray native-exp P/PB source/reserve report; preserves transmission and local band tau."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np

GROUPS=['old_prefix_supervised','new_prefix_supervised','old_local_reserved','new_whole_patch_reserved','duplicate_excluded']
PROFILES=['native','halo1','all_cells','uniform','uniform2x']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def inspect_trace(folder,row,profile):
    path=folder/row['profiles'][profile]['trace']
    with np.load(path,allow_pickle=False)as z:
        start,end,sigma,weight,trans=[z[k].reshape(-1).astype(np.float64)for k in ['t_start','t_end','sigma','weights','transmittance']]
    point=np.array(row['pixel']['witness_xyz']);origin=np.array(row['origin']);direction=np.array(row['direction']);depth=float((point-origin)@direction);mid=(start+end)*.5
    mass=[float(weight[mid<depth-.02].sum()),float(weight[abs(mid-depth)<=.02].sum()),float(weight[mid>depth+.02].sum()),max(0.,1.-float(weight.sum()))]
    if abs(sum(mass)-1)>1e-5:raise ValueError('Termination/transmission do not conserve probability')
    half=max(.005,.01*depth);lo=np.maximum(start,depth-half);hi=np.minimum(end,depth+half);length=np.maximum(hi-lo,0.);tau=float((sigma*length).sum())
    # Partial intervals use transmittance at the actual clipped interval start.
    band_mass=float((trans*np.exp(-sigma*np.maximum(lo-start,0.))*(-np.expm1(-sigma*length))).sum())
    return dict(mass=mass,positive_band_halfwidth=half,positive_band_tau=tau,
                positive_band_local_opacity=float(-np.expm1(-tau)),positive_band_unconditional_mass=band_mass,
                trace_sha256=sha(path))

def summarize(rows):
    result=dict(rays=len(rows),points=len({r['point']for r in rows}),profiles={})
    for profile in PROFILES:
        values=[r['profiles'][profile]for r in rows]
        result['profiles'][profile]=dict(mean_mass=np.mean([v['mass']for v in values],axis=0).tolist(),
            mean_positive_band_tau=float(np.mean([v['positive_band_tau']for v in values])),
            mean_positive_band_local_opacity=float(np.mean([v['positive_band_local_opacity']for v in values])),
            mean_positive_band_unconditional_mass=float(np.mean([v['positive_band_unconditional_mass']for v in values])))
    result['max_mass_component_8192_16384_delta']=max(r['mass_convergence_delta']for r in rows)
    result['max_positive_band_mass_8192_16384_delta']=max(r['band_mass_convergence_delta']for r in rows)
    result['converged_rgb_opacity_rays']=sum(r['rgb_opacity_converged']for r in rows)
    return result

def compare(campaign,output):
    c=Path(campaign);a=c/'independent_occupancy';query=a/'queries/clothing_103_predeclared.json';q=json.loads(query.read_text());output=Path(output);output.mkdir(exist_ok=True,parents=True)
    expected=[(p['witness_id'],p['camera_name'],p['x'],p['y'],p['diagnostic_group'])for p in q['pixels']]
    sources={'exp_reference':a/'structural_exp_hplus1_015188','P_prefix':a/'exp_geometry_P_015188','PB_prefix_band':a/'exp_geometry_PB_015188'};models={};files={str(query):sha(query)}
    for name,root in sources.items():
        folder=root/'clothing103_trace/rays';audit=json.loads((folder/'audit.json').read_text());chain=json.loads((root/'clothing103_trace/coordinate_chain.json').read_text());health=json.loads((root/'clothing103_producer_health.json').read_text())
        if audit['step']!=15188 or not audit['frozen_state_unchanged']or not health['audit_environment_healthy']:raise ValueError('Matched immutable checkpoint and healthy producer required')
        ids=[(p['pixel']['witness_id'],p['pixel']['camera_name'],p['pixel']['x'],p['pixel']['y'],p['pixel']['diagnostic_group'])for p in audit['rows']]
        if ids!=expected:raise ValueError('Queries differ from predeclared103 source observations')
        if chain['queries_sha256']!=sha(query)or chain['correct_sh_directions']is not True or chain['density_activation']!='trunc_exp' or chain['feature_reweighting_strength']!=1.:raise ValueError('Actual source/field flags differ')
        rows=[]
        for row in audit['rows']:
            pixel=row['pixel'];profiles={p:inspect_trace(folder,row,p)for p in PROFILES};v=row['uniform_convergence'];conv=v['opacity_abs_difference']<=.005 and v['foreground_rgb_max_abs_difference']<=1/255
            rows.append(dict(point=pixel['witness_id'],camera=pixel['camera_name'],actor=pixel['actor'],region=pixel['region'],group=pixel['diagnostic_group'],profiles=profiles,
                rgb_opacity_converged=bool(conv),original_uniform_convergence=v,mass_convergence_delta=float(np.max(abs(np.array(profiles['uniform']['mass'])-profiles['uniform2x']['mass']))),
                band_mass_convergence_delta=abs(profiles['uniform']['positive_band_unconditional_mass']-profiles['uniform2x']['positive_band_unconditional_mass'])))
        groups={group:summarize([r for r in rows if r['group']==group])for group in GROUPS}
        actors={actor:summarize([r for r in rows if r['actor']==actor and r['group']!='duplicate_excluded'])for actor in ['black','grey']}
        models[name]=dict(rows=rows,groups=groups,actors=actors,step=15188,checkpoint_sha256=audit['checkpoint_sha256'],source=str(root),coordinate_chain=chain,producer_health=health)
        if (root/'renders/metrics.json').exists():
            metrics=json.loads((root/'renders/metrics.json').read_text());models[name]['heldout_metrics']={k:metrics['metrics'][k]for k in ['psnr','ssim','lpips']};models[name]['heldout_by_actor']={actor:{k:v[k]for k in ['psnr','ssim','lpips']}for actor,v in metrics['by_actor'].items()}
        for source in [folder/'audit.json',root/'clothing103_trace/coordinate_chain.json',root/'clothing103_producer_health.json']:
            files[str(source)]=sha(source)
    result=dict(kind='MATCHED15188_PREDECLARED103_CLOTHING_DIAGNOSTIC',models=models,query_sha256=sha(query),mass_order=['before_.02_band','inside_.02_band','after_.02_band','transmission'],
        convergence_policy='opacity abs difference<=.005 and foreground RGB max abs difference<=1/255; full raw values preserved',
        positive_band_policy='max(.005,.01*projected witness depth), exact interval clipping; diagnostic pixel-ray integral, not training-loss replay',
        source_groups='Names describe predeclared source roles. Old18 and new66 are auxiliary-fitted in P/PB; exp_reference has no auxiliary labels; all are TRAIN RGB evidence.',
        caveats=['Reserved whole patches are excluded from auxiliary objectives but not RGB training.','Density in skipped cells is not independently certified occupancy error.','Local band opacity is conditional on entering that interval; unconditional mass also accounts for foreground occlusion.','No complete-scene occupancy certificate or automatic quality promotion.'],files=files)
    (output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for ax,group in zip(axes.flat,GROUPS[:4]):
        values=np.array([models[name]['groups'][group]['profiles']['native']['mean_mass']for name in models])*100;base=np.zeros(len(models))
        for i,(label,color)in enumerate(zip(['before','witness ±.02','after','transmission'],['#d86147','#48a876','#5784bb','#c8c8c8'])):
            ax.bar(np.arange(len(models)),values[:,i],bottom=base,label=label,color=color);base+=values[:,i]
        ax.set_xticks(np.arange(len(models)),['exp reference','P prefix84','PB +band']);ax.set_title(group.replace('_',' '));ax.set_ylim(0,104);ax.set_ylabel('Native unconditional mass (%)');ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(ncol=4,fontsize=8);fig.suptitle('Matched15188: predeclared clothing sources and whole-patch reserves');fig.savefig(output/'native_mass.png',dpi=150);plt.close(fig)
    lines=['# Native-exp P/PB matched15188 clothing diagnostic','', 'This is a source-based geometric diagnostic, not an occupancy or quality approval. All models use the same103 predeclared TRAIN pixel rays.','', '| Group | Model | Native before/band/after/transmission (%) | Native positive-band local opacity | Dense positive-band unconditional mass |','|---|---|---|---:|---:|']
    for group in GROUPS[:4]:
        for name,m in models.items():
            r=m['groups'][group];n=r['profiles']['native'];dense=r['profiles']['uniform2x'];lines.append('| '+group+' | '+name+' | '+' / '.join(f'{100*x:.2f}'for x in n['mean_mass'])+f" | {n['mean_positive_band_local_opacity']:.5f} | {dense['mean_positive_band_unconditional_mass']:.5f} |")
    lines += ['', 'Local opacity may improve while front density still absorbs rays; consult unconditional mass and the separate native/halo/dense profiles. Reserved groups are TRAIN RGB evidence, excluded only from geometry labels.','', 'All producer before/after identities and immutable-model receipts are included in comparison.json. Actual held-out PSNR/SSIM/LPIPS are included where the staged metrics exist. No optimizer update or training callback was run by the auditor.']
    (output/'report.md').write_text('\n'.join(lines)+'\n');(output/'complete.json').write_text(json.dumps(dict(complete=True,models=list(models),query_sha256=sha(query),comparison_sha256=sha(output/'comparison.json'),production_mutated=False),indent=2)+'\n');print(output,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--wait-state',type=Path);a=p.parse_args()
    if a.wait_state:
        while True:
            state=json.loads(a.wait_state.read_text())if a.wait_state.exists()else{}
            if state.get('phase')=='traces_complete':break
            if state.get('phase')=='audit_requires_attention':raise RuntimeError('Audit requires attention; not a training failure claim')
            time.sleep(60)
    compare(a.campaign,a.output)
