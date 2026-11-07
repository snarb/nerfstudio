"""Saved actual native/dense replay diagnostics; no GPU, network or label changes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from summarize_marshawn_depth_residuals import quantile_depth,stats

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(folder,row,profile):
 p=folder/row['profiles'][profile]['trace']
 with np.load(p)as z:
  start,end,sigma,weight,trans=[z[k].reshape(-1).astype(float)for k in ['t_start','t_end','sigma','actual_weights','transmittance']];rgb=z['rgb'].astype(float)
 mid=(start+end)/2;alpha=float(weight.sum());result=dict(samples=len(start),native_cap_hit=len(start)>=1024 if profile=='native'else None,opacity=alpha,transmission=max(0.,1.-alpha),RGB=(weight[:,None]*rgb).sum(0).tolist(),trace_sha256=sha(p))
 if 'witness_xyz'in row['pixel']:
  depth=float((np.array(row['pixel']['witness_xyz'])-row['origin'])@row['direction']);mass=[float(weight[mid<depth-.02].sum()),float(weight[abs(mid-depth)<=.02].sum()),float(weight[mid>depth+.02].sum()),max(0.,1.-alpha)]
  q={str(v):quantile_depth(start,end,sigma,weight,trans,v)for v in [.1,.5,.9]};result.update(witness_depth=depth,mass_before_band_after_transmission=mass,signed_t50_minus_witness=None if q['0.5']is None else q['0.5']-depth,width10_90=None if q['0.1']is None or q['0.9']is None else q['0.9']-q['0.1'])
 return result

def summarize(rows):
 if not rows:return None
 result=dict(rays=len(rows),profiles={})
 for profile in ['native','uniform','uniform2x']:
  v=[r['profiles'][profile]for r in rows];result['profiles'][profile]=dict(mean_opacity=float(np.mean([x['opacity']for x in v])),maximum_samples=max(x['samples']for x in v),cap_hits=sum(bool(x['native_cap_hit'])for x in v))
  if 'mass_before_band_after_transmission'in v[0]:result['profiles'][profile].update(mean_mass=np.mean([x['mass_before_band_after_transmission']for x in v],0).tolist(),signed_t50_minus_witness=stats([x['signed_t50_minus_witness']for x in v]),width10_90=stats([x['width10_90']for x in v]))
 result['convergence']=dict(max_opacity_delta=max(r['convergence']['opacity_abs_difference']for r in rows),max_RGB_delta=max(r['convergence']['foreground_rgb_max_abs_difference']for r in rows),passed_rays=sum(r['convergence']['opacity_abs_difference']<=.005 and r['convergence']['foreground_rgb_max_abs_difference']<=1/255 for r in rows))
 return result

def main(paths,out):
 models={};bound={}
 for value in paths:
  name,root=value.split('=',1);root=Path(root);groups={};allrows={}
  for label in ['clothing103','DC12','contour9']:
   folder=root/(label+'_trace/rays');auditpath=folder/'audit.json';audit=json.loads(auditpath.read_text());assert audit['frozen_state_unchanged'];bound[str(auditpath)]=sha(auditpath);rows=[]
   for item in audit['rows']:
    pixel=item['pixel'];rows.append(dict(pixel=pixel,profiles={p:record(folder,item,p)for p in ['native','uniform','uniform2x']},convergence=item['uniform_convergence']))
   allrows[label]=rows;groups[label]=summarize(rows)
   if label=='clothing103':
    for key in ['actor','diagnostic_group']:
     for category in sorted({r['pixel'][key]for r in rows}):groups[label+'/'+key+'/'+category]=summarize([r for r in rows if r['pixel'][key]==category])
  renders=list((root/'renders').glob('step-*/metrics.json'));assert len(renders)==1;metrics=json.loads(renders[0].read_text());bound[str(renders[0])]=sha(renders[0]);models[name]=dict(source=str(root),step=audit['step'],checkpoint_sha256=audit['checkpoint_sha256'],metrics={k:v for k,v in metrics['metrics'].items()if k!='validation_loss'},by_actor={a:{k:v for k,v in m.items()if k!='validation_loss'}for a,m in metrics['by_actor'].items()},groups=groups,rows=allrows)
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(dict(models=models,source_hashes=bound,no_GPU=True,actual_native_weight_used=True,no_opacity_renormalization=True,quality_promoted=False,limitations=['Witnesses are sparse TRAIN evidence, not complete geometry truth.','5568 inherited hair/chest ambiguity remains; diagnostic band is not alpha1 target.','DC12 and contour9 are EVAL diagnostics, never fitting targets.','Dense samples query skipped space; its sigma is not certified occupancy truth.','Termination quantiles are unconditional; missing quantiles stay unknown.']),indent=2)+'\n');print(out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.model,a.output)
