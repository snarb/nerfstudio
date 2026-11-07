"""Inspect proposed RGB profiles and independently reconstruct unchanged body80 E_q."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def counts(n,step):
    ramp=.5*min(1.,max(0.,(step-4096)/4096));p=np.array([2/3*(1-ramp),2/3*ramp,2/15,1/5]);want=p*n;out=np.floor(want).astype(int);out[np.argsort(-(want-out))[:n-out.sum()]]+=1;return out


def components(w,sem,levels):
    core=sem['core']&sem['valid'];band=sem['band']&sem['valid'];valid=sem['valid'];union=sem['black']|sem['grey'];h,ww=core.shape
    means=[]
    for mask in [core,band,valid]:
        if not mask.any():mask=core if core.any()else valid
        means.append(float(w[mask].mean(dtype=np.float64)))
    def cells(x):return x[:levels.shape[0]*8,:levels.shape[1]*8].reshape(levels.shape[0],8,levels.shape[1],8).transpose(0,2,1,3)
    cm=cells(core);um=cells(union);eligible=(um.mean((-2,-1))>=.9)&cm.any((-2,-1));member=cm&um;denom=member.sum((-2,-1));eligible&=denom>0
    num=(cells(w).astype(np.float64)*member).sum((-2,-1));percell=np.divide(num,denom,out=np.zeros_like(num),where=denom>0);prob=np.array([(eligible&(levels==l)).any()for l in range(16)],float)*np.linspace(1,3,16)
    fas=means[0]
    if prob.sum():
        prob/=prob.sum();fas=sum(prob[l]*float(percell[eligible&(levels==l)].mean())for l in range(16)if prob[l])
    return np.array([means[0],fas,means[1],means[2]])


def main(c,out,levels_path,bindings_path):
    summary=json.loads((out/'summary.json').read_text());levelz=np.load(levels_path);bindings=json.loads(bindings_path.read_text());means={k:[]for k in ['original','bag6522','delta2','delta4']};aggregate={k:{}for k in means};profiles={};(out/'transfer').mkdir(exist_ok=True)
    for row in summary['rows']:
        name=row['camera'];freq=c/'data/lookcloser_frequencies'/f'train_{name}.pt';assert sha(freq)==bindings[name]['frequency_sha256'];assert sha(freq.with_suffix('.json'))==bindings[name]['metadata_sha256'];assert bindings[name]['patch_size']==bindings[name]['stride']==8
        with np.load(c/'data/semantics'/f'{name}.npz')as z:sem={k:z[k]for k in ['W64','core','band','valid','black','grey']}
        p=np.load(out/'profiles'/f'{name}.npz');weights={'original':sem['W64'],'bag6522':p['weight_bag6522'],'delta2':p['weight_delta2'],'delta4':p['weight_delta4']}
        for k,w in weights.items():
            means[k].append(components(w,sem,levelz[name]))
            for group,s in row['statistics'][k].items():
                total=aggregate[k].setdefault(group,dict(pixels=0,sum_weight=0.));total['pixels']+=s['pixels'];total['sum_weight']+=s['sum_weight']
        np.savez_compressed(out/'transfer'/f'{name}.npz',delta2=weights['delta2'],delta4=weights['delta4'])
        profiles[name]=dict(path=str(out/'profiles'/f'{name}.npz'),archive_sha256=sha(out/'profiles'/f'{name}.npz'),proposal_components={k:means[k][-1].tolist()for k in means})
    phases=[]
    for step in [0,4096,6144,8192,111316]:
        quota=counts(4096,step);norm={k:float(np.mean(means[k],axis=0)@(quota/4096))for k in means};phases.append(dict(step=step,quotas=quota.tolist(),normalizer=norm,inside_gradient_scale_vs_original={k:norm['original']/v for k,v in norm.items()}))
    actual=json.loads((c/'semantic_actor_bagloss_prepared_20261106/runtime_cpu_proof.json').read_text());checks=[]
    for phase in actual['phases']:
        ours=next((r for r in phases if r['step']==phase['step']),None)
        if ours is None:ours=next(r for r in phases if r['step']==8192)
        errors=[abs(ours['normalizer']['original']-phase['reference_normalizer']),abs(ours['normalizer']['bag6522']-phase['corrected_normalizer'])];assert max(errors)<1e-12,errors;checks.append(dict(step=phase['step'],max_error=max(errors)))
    samples=[]
    coverage=c/'independent_occupancy/quarter111316_artifact_source_audit_v1/source_projection/coverage.json';cov=json.loads(coverage.read_text())
    for name in profiles:
        with np.load(profiles[name]['path'])as z:p={k:z[k]for k in ['distance_HD','inside_RAW_union','weight_bag6522','weight_delta2','weight_delta4']}
        for point in cov['points']:
            if point['kind']!='q50':continue
            r=next(r for r in point['train_projections']if r['camera']==name)
            if not r['in_frame']:continue
            x,y=r['pixel'];samples.append(dict(artifact=point['label'],camera=name,pixel=[x,y],distance_HD=float(p['distance_HD'][y,x]),inside_RAW_union=bool(p['inside_RAW_union'][y,x]),original_W64=r['W64'],bag6522=float(p['weight_bag6522'][y,x]),delta2=float(p['weight_delta2'][y,x]),delta4=float(p['weight_delta4'][y,x]),UNKNOWN=r['UNKNOWN']))
    hair=[];queries=c/'semantic_actor_candidate_v2_20261106/source/TRAIN103_queries.json';q=json.loads(queries.read_text())
    for row in q['pixels']:
        if row.get('witness_id')!=5568:continue
        p=np.load(profiles[row['camera_name']]['path']);x,y=row['x'],row['y'];hair.append(dict(query=row,inside_RAW_union=bool(p['inside_RAW_union'][y,x]),distance_HD=float(p['distance_HD'][y,x]),bag6522=float(p['weight_bag6522'][y,x]),delta2=float(p['weight_delta2'][y,x]),delta4=float(p['weight_delta4'][y,x])))
    result=dict(kind='INDEPENDENT_CPU_BODY80_EXPECTATION_AND_RGB_PROFILE_EFFECTS',source_summary_sha256=sha(out/'summary.json'),frequency_levels_sha256=sha(levels_path),frequency_bindings_sha256=sha(bindings_path),actual_reference_proof_sha256=sha(c/'semantic_actor_bagloss_prepared_20261106/runtime_cpu_proof.json'),original_and_bag_expectations_match_actual63_adapter=checks,phases=phases,aggregate=aggregate,cameras=profiles,artifact_source_samples=samples,hair_boundary5568=hair,analytic_curve=[dict(distance_HD=d,delta2=.1+.9*np.exp(-d*d/8),delta4=.1+.9*np.exp(-d*d/32))for d in [0,.5,1,2,4,8,16]],no_source_mutation=True,training_labels_generated=False,temporal_claim=False)
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(phases=phases,reference_checks=checks,hair=hair),indent=2))


def sampled_groups(c,out,levels_path):
    analysis=json.loads((out/'analysis.json').read_text());levels=np.load(levels_path);means={k:[]for k in ['original','bag6522','delta2','delta4']}
    for name in analysis['cameras']:
        with np.load(c/'data/semantics'/f'{name}.npz')as z:sem={k:z[k]for k in ['W64','core','band','valid','black','grey']}
        with np.load(out/'profiles'/f'{name}.npz')as z:p={k:z[k]for k in ['inside_RAW_union','weight_bag6522','weight_delta2','weight_delta4']}
        weights={'original':sem['W64'],'bag6522':p['weight_bag6522'],'delta2':p['weight_delta2'],'delta4':p['weight_delta4']}
        for k,w in weights.items():means[k].append(components(w*~p['inside_RAW_union'],sem,levels[name]))
    rows=[]
    for phase in analysis['phases']:
        q=np.array(phase['quotas'])/4096;values={k:float(np.mean(v,axis=0)@q)for k,v in means.items()};rows.append(dict(step=phase['step'],outside_Eq_weight=values,outside_fraction_of_normalized_RGB_objective={k:v/phase['normalizer'][k]for k,v in values.items()}))
    (out/'sampled_outside_weight.json').write_text(json.dumps(dict(analysis_sha256=sha(out/'analysis.json'),phases=rows,meaning='Expected weighted RGB contribution of outside original RAW union under unchanged body80 proposal; not an opacity or empty-space label.'),indent=2)+'\n');print(json.dumps(rows,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--levels',type=Path,required=True);ap.add_argument('--bindings',type=Path);ap.add_argument('--sampled-only',action='store_true');a=ap.parse_args()
    if a.sampled_only:sampled_groups(a.campaign,a.output,a.levels)
    else:main(a.campaign,a.output,a.levels,a.bindings)
