"""CPU summary of frozen focused cap1024/2048/restored1024; dense field must stay identical."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def main(root,helpers):
 if not json.loads((root/'complete.json').read_text())['passed']:raise ValueError('Both restored source probes must pass')
 compare=load('_FR_cloth_comparison',helpers/'compare_selected_traces.py');sys.path.insert(0,str(helpers));sampling=load('_FR_sampling',helpers/'audit_marshawn_saved_sampling.py')
 result=dict(kind='SAME_STATE_LEGACY_CAP_COUNTERFACTUAL',models={},GPU_used=False,optimizer_updates=0,quality_promotion=False,limits=['Fixed occupancy and frequency grids; no adaptation or refreshed occupancy.','All-cell/uniform density remains constrained by fixed semantic admission and is not independent physical geometry.','TRAIN-source local reserves are not held-out RGB; DC12 has no physical depth truth.'])
 for name in ['focus_FR1','focus_FR03']:
  arm=root/name;proof=json.loads((arm/'complete.json').read_text());phases={}
  for phase in ['cap1024','cap2048','cap1024_restored']:
   folder=arm/phase;trace=folder/'clothing103_trace/rays';audit=json.loads((trace/'audit.json').read_text());scores=json.loads((folder/'renders/step-000075940/metrics.json').read_text());groups={}
   for group in compare.GROUPS:
    rows=[r for r in audit['rows']if r['pixel']['diagnostic_group']==group];profiles={}
    for profile in compare.PROFILES:
     items=[compare.inspect_trace(trace,r,profile)for r in rows]
     profiles[profile]=dict(mean_mass=np.mean([r['mass']for r in items],axis=0).tolist(),positive_band_local_opacity=float(np.mean([r['positive_band_local_opacity']for r in items])))
    groups[group]=dict(count=len(rows),profiles=profiles)
   phases[phase]=dict(metrics={k:scores['metrics'][k]for k in ['psnr','ssim','lpips']},by_actor={k:{v:x[v]for v in ['psnr','ssim','lpips']}for k,x in scores['by_actor'].items()},by_view={k:{v:x[v]for v in ['psnr','ssim','lpips']}for k,x in scores['by_view'].items()},groups=groups,sampling={label:sampling.inspect(folder,label)['summary']for label in ['clothing103','DC12']},proof_sha256=sha(arm/'complete.json'))
  for phase,item in phases.items():
   for label,stats in item['sampling'].items():
    for profile in ['native','halo1','all_cells']:
     stats[profile]['at_original_1024_count']=stats[profile].pop('at_cap');stats[profile]['actual_cap']=2048 if phase=='cap2048' else 1024;stats[profile]['at_cap']=stats[profile]['saturated']
  density_changes={}
  for label in ['clothing103','DC12']:
   first=arm/'cap1024'/(label+'_trace/rays');other=arm/'cap2048'/(label+'_trace/rays');rows=[]
   for file in sorted(first.glob('pixel*_uniform2x.npz')):
    with np.load(file)as a,np.load(other/file.name)as b:
     for k in ['t_start','t_end','xyz','sigma','rgb','weights','actual_weights']:
      if not np.array_equal(a[k],b[k]):raise ValueError('Sampler-cap perturbation changed the dense field/reference')
     before=a['sigma'].reshape(-1).astype(float);after=b['sigma'].reshape(-1).astype(float);delta=abs(after-before)
     rows.append(dict(ray=file.name,positive_sigma_mean_before=float(before.mean()),positive_sigma_mean_after=float(after.mean()),sigma_maxabs=float(delta.max(initial=0)),sigma_changed_fraction=float((delta>1e-6).mean()),opacity_before=float(a['weights'].sum()),opacity_after=float(b['weights'].sum())))
   density_changes[label]=rows
  result['models'][name]=dict(source_checkpoint_sha256=proof['source_checkpoint_sha256'],source_step=75940,phases=phases,whole_interval_same_position_density_change=density_changes,restoration_proof=proof)
 (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# Same-state focused sampler-cap counterfactual','','No optimizer step, frequency refresh or occupancy update. Both fixed-FR arms use the same immutable focused75940 state. Only max_steps_per_ray1024→2048 changes during a phase; dense arrays must remain bitwise unchanged.','','| Model | Cap phase | PSNR | SSIM | LPIPS |','|---|---:|---:|---:|---:|']
 for name,model in result['models'].items():
  for phase in ['cap1024','cap2048','cap1024_restored']:
   x=model['phases'][phase]['metrics'];lines.append(f"| {name} | {phase} | {x['psnr']:.6f} | {x['ssim']:.6f} | {x['lpips']:.8f} |")
 lines+=['','| Model | Group | Native band cap1024% | Native band cap2048% | Dense band cap1024% | Dense band cap2048% |','|---|---|---:|---:|---:|---:|']
 for name,model in result['models'].items():
  for group in compare.GROUPS[:4]:
   vals=[model['phases'][phase]['groups'][group]['profiles'][prof]['mean_mass'][1]*100 for prof in ['native','uniform2x']for phase in ['cap1024','cap2048']];lines.append('| '+name+' | '+group+' | '+' | '.join(f'{x:.4f}'for x in vals)+' |')
 lines+=['','Sampler-cap perturbation is an integration diagnostic with learned field/occupancy/frequency held fixed. Dense arrays remain exactly unchanged. It does not establish the quality of training with a larger cap; all-cells changes are separate from actual native traversal.','', 'All seven serialized states, actual pipeline/global RNG and private focus state are unchanged. Nonpersistent feature LUT is restored. Original1024 native115 ray arrays restore bitwise; FHD RGB8/alpha16 images use pre-existing max1-code and changed fractions1e-5/.001, all6/view/actor metrics1e-5. Frozen source uses the successful v3 same-state FR baseline; no training occurred.']
 (root/'report.md').write_text('\n'.join(lines)+'\n');print(root/'comparison.json')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--helpers',type=Path,required=True);a=p.parse_args();main(a.root,a.helpers)
