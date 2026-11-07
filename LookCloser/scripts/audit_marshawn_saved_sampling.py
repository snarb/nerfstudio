"""CPU-only native sampler/cap/integration diagnostics from immutable trace arrays."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from summarize_marshawn_depth_residuals import quantile_depth

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(values):
 a=np.array([x for x in values if x is not None],float)
 return dict(n=len(a),unknown=len(values)-len(a),mean=float(a.mean())if len(a)else None,median=float(np.median(a))if len(a)else None,p90=float(np.quantile(a,.9))if len(a)else None,max=float(a.max())if len(a)else None)
def inspect(root,label):
 folder=root/(label+'_trace/rays');path=folder/'audit.json';audit=json.loads(path.read_text());rows=[]
 for record in audit['rows']:
  profiles={};arrays={}
  for name in ['native','halo1','all_cells','uniform','uniform2x']:
   info=record['profiles'][name];p=folder/info['trace']
   with np.load(p,allow_pickle=False)as src:z={k:src[k].reshape(-1).astype(float)for k in ['t_start','t_end','sigma','weights','transmittance']}
   arrays[name]=z;start,end=z['t_start'],z['t_end'];dt=end-start
   if len(start)!=info['samples']or(np.diff(start)<-1e-6).any():raise ValueError('Trace order/count mismatch')
   t50=quantile_depth(*[z[k]for k in ['t_start','t_end','sigma','weights','transmittance']],.5)
   profiles[name]=dict(samples=len(start),recorded_saturation=info['saturation'],at_1024_cap=len(start)==1024,interval_width=stats(dt.tolist()),last_end=float(end.max())if len(end)else None,median_depth=t50,opacity=info['opacity'],rgb=info['integrated_foreground_rgb'],trace_sha256=sha(p))
  n=profiles['native'];d=profiles['uniform2x'];dense=arrays['uniform2x'];mid=(dense['t_start']+dense['t_end'])/2;last=n['last_end'];tail=float(dense['weights'][mid>last].sum())if last is not None else float(dense['weights'].sum())
  r=dict(pixel=record['pixel'],profiles=profiles,native_dense_RGB_maxabs=float(np.max(abs(np.array(n['rgb'])-d['rgb']))),native_dense_opacity_abs=abs(n['opacity']-d['opacity']),native_dense_t50_abs=None if n['median_depth']is None or d['median_depth']is None else abs(n['median_depth']-d['median_depth']),allcells_dense_RGB_maxabs=float(np.max(abs(np.array(profiles['all_cells']['rgb'])-d['rgb']))),native_halo_RGB_maxabs=float(np.max(abs(np.array(n['rgb'])-profiles['halo1']['rgb']))),dense_weight_after_native_last_end=tail,dense_converged=record['uniform_convergence']['opacity_abs_difference']<=.005 and record['uniform_convergence']['foreground_rgb_max_abs_difference']<=1/255)
  rows.append(r)
 summary={}
 for name in ['native','halo1','all_cells']:
  vals=[r['profiles'][name]for r in rows];summary[name]=dict(samples=stats([v['samples']for v in vals]),saturated=sum(v['recorded_saturation']>0 for v in vals),at_cap=sum(v['at_1024_cap']for v in vals),max_interval=stats([v['interval_width']['max']for v in vals]))
 for key in ['native_dense_RGB_maxabs','native_dense_opacity_abs','native_dense_t50_abs','allcells_dense_RGB_maxabs','native_halo_RGB_maxabs','dense_weight_after_native_last_end']:summary[key]=stats([r[key]for r in rows])
 summary['native_dense_RGB_over_1code']=sum(r['native_dense_RGB_maxabs']>1/255 for r in rows);summary['dense_converged']=sum(r['dense_converged']for r in rows)
 return dict(rows=rows,summary=summary,checkpoint_sha256=audit['checkpoint_sha256'],audit_sha256=sha(path),step=audit['step'])
def main(root,out):
 result=dict(kind='SAVED_NATIVE_CAP_AND_INTEGRATION_DIAGNOSTIC',source=str(root),profiles={x:inspect(root,x)for x in ['clothing103','DC12']},GPU_used=False,optimizer_updates=0,caveats=['Uniform bypasses actual occupancy/cap; discrepancy is not automatically coarse-integration error.','All-cells ARM can itself saturate; compare recorded saturation separately.','Skipped-space density is not physical geometry truth.','One calibrated ray per probe; no full-scene or training-batch performance claim.'])
 out.mkdir(parents=True,exist_ok=True);(out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v['summary']for k,v in result['profiles'].items()}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.source,a.output)
