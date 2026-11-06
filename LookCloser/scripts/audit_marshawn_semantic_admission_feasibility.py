"""CPU-only whole-LC-cell semantic admission feasibility; no producer mutation."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter
from marshawn_semantic_cell_admission import WholeCellAdmission


def sha(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')


def main(c,source,out):
    out.mkdir(exist_ok=False,parents=True);began=time.time();record=json.loads((source/'grid_receipt.json').read_text());ready=json.loads((c/'data/ready.json').read_text())
    if sha(source/'semantic_grid.npz')!=record['grid_sha256']:raise ValueError('Reviewed semantic grid changed')
    if record.get('no_eval_masks_or_RGB_read')is not True or record['native_pixel_support_width']!=64 or len(record['train_cameras'])!=63 or set(record['train_cameras'])&{'DC','HC','KA'}:raise ValueError('TRAIN-only W64 source required')
    if record['transforms_sha256']!=sha(c/'data/transforms.json')or record['source_ready_sha256']!=sha(c/'data/ready.json'):raise ValueError('Prepared source/frame changed')
    with np.load(source/'semantic_grid.npz')as z:
        gate=z['semantic_gate_pad1'].copy();base=z['semantic_gate0'].copy();bounds=z['aabb'].copy();transform=z['world_to_box'].copy()
    assert gate.shape==(128,128,128)and np.array_equal(gate,maximum_filter(base,size=3,mode='constant',cval=0))
    np.testing.assert_allclose(bounds,ready['aabb'],rtol=0,atol=1e-12);np.testing.assert_allclose(transform,ready['world_to_box'],rtol=0,atol=1e-12)
    actor=c/'independent_occupancy/exp_followup_actor_015188';receipt=json.loads((actor/'step-000015188_grid.json').read_text());archive=actor/'step-000015188_grid.npz'
    if sha(archive)!=receipt['grid_archive_sha256']or receipt['radius']!=0:raise ValueError('Expected immutable actual actorR0 grid')
    with np.load(archive)as z:active=z['active'].copy();core=z['core'].copy();boxes=z['cascade_aabbs'].copy();resolution=z['resolution'].copy()
    assert json.loads((actor/'archive_receipt.json').read_text())['checkpoint_sha256']==receipt['checkpoint_sha256']
    assert json.loads((actor/'clothing103_trace/rays/audit.json').read_text())['checkpoint_sha256']==receipt['checkpoint_sha256']
    assert len(active)==1 and np.array_equal(active,core)and tuple(active.shape[1:])==tuple(resolution)
    np.testing.assert_allclose(boxes[0].reshape(2,3),bounds,rtol=0,atol=2e-7)
    semantic=WholeCellAdmission(gate,bounds);mapped=semantic.map_target_grid(resolution,boxes[0].reshape(2,3));cover=WholeCellAdmission(mapped,boxes[0].reshape(2,3))
    # Independent source-center containment check, in the already prepared floor
    # frame. world_to_box is recorded, deliberately NOT applied a second time.
    centers=bounds[0]+(np.argwhere(gate)+.5)/128*(bounds[1]-bounds[0]);assert cover.point_admitted(centers).all()
    np.savez_compressed(out/'hypothetical_admission_only_NOT_APPLIED.npz',admitted=mapped[None],cascade_aabbs=boxes,resolution=resolution)
    qpath=c/'exp_geometry_candidate_20261106/clothing_103_predeclared.json';q=json.loads(qpath.read_text());assert sha(qpath)=='e11c111e4a41b02417e98ae2e0f4f5f6055883c070539b2334748974bb3c712c'
    witness_rows=[]
    for p,r in zip(q['pixels'],q['rays']):
        xyz=np.array(p['witness_xyz']);o=np.array(r['origin']);d=np.array(r['direction']);t=float((xyz-o)@d)
        witness_rows.append(dict(point=p['witness_id'],camera=p['camera_name'],group=p['diagnostic_group'],xyz_retained=bool(cover.point_admitted(xyz[None])[0]),pixel_ray_at_depth_retained=bool(cover.point_admitted((o+d*t)[None])[0])))
    wr=json.loads((source/'witness_receipt.json').read_text());anchors=[r for r in wr['anchors']if r['name']in {'toe_upper','sfm_6862'}]
    assert len(anchors)==2
    toe_rows=[dict(name=r['name'],label=r['label'],xyz_floor=r['xyz_floor'],retained=bool(cover.point_admitted(np.array(r['xyz_floor'])[None])[0]))for r in anchors]
    def analyze_trace(folder,kind):
        audit_path=folder/'rays/audit.json';audit=json.loads(audit_path.read_text());rows=[]
        for row in audit['rows']:
            origin,direction=np.array(row['origin']),np.array(row['direction']);p=row['pixel']
            with np.load(folder/'rays'/row['profiles']['native']['trace'])as z:
                start,end,weights=[z[k].reshape(-1)for k in ['t_start','t_end','weights']]
            a=origin+start[:,None]*direction;b=origin+end[:,None]*direction;mid=(a+b)*.5
            keep_mid=cover.point_admitted(mid);keep_whole=cover.boxes_admitted(np.minimum(a,b),np.maximum(a,b))
            result=dict(pixel=p,midpoint_cell_excluded_mass=float(weights[~keep_mid].sum()),definitely_excluded_full_interval_mass=float(weights[~keep_whole].sum()),native_opacity=float(weights.sum()))
            if 'witness_xyz'in p:
                depth=float((np.array(p['witness_xyz'])-origin)@direction);early=(start+end)*.5<depth-.02;band=abs((start+end)*.5-depth)<=.02
                result.update(early_mass=float(weights[early].sum()),early_mass_in_excluded_midpoint_cells=float(weights[early&~keep_mid].sum()),early_mass_in_definitely_excluded_intervals=float(weights[early&~keep_whole].sum()),band_mass=float(weights[band].sum()),band_mass_in_definitely_excluded_intervals=float(weights[band&~keep_whole].sum()))
            rows.append(result)
        groups={}
        for group in sorted({r['pixel']['diagnostic_group']for r in rows}):
            selected=[r for r in rows if r['pixel']['diagnostic_group']==group]
            groups[group]={k:float(np.mean([r[k]for r in selected]))for k in selected[0]if k!='pixel'};groups[group]['rays']=len(selected)
        return dict(kind=kind,rows=rows,groups=groups,source_trace_audit_sha256=sha(audit_path))
    traces={'actor_TRAIN103':analyze_trace(actor/'clothing103_trace','TRAIN_SOURCE_DIAGNOSTIC'),
            'actor_EVAL_DC12':analyze_trace(actor/'DC12_trace','READ_ONLY_EVAL_DIAGNOSTIC_NOT_GATE_INPUT')}
    result=dict(kind='CPU_CONSERVATIVE_WHOLE_LC_CELL_SEMANTIC_ADMISSION_FEASIBILITY',coordinate_frame='accepted_floor_world',coordinate_transform_applied=False,
        semantic_source_sha256=record['grid_sha256'],source_receipt_sha256=sha(source/'grid_receipt.json'),source_whole_train_ray_receipt_sha256=sha(source/'final_train_ray_receipt.json'),
        source_checkpoint_sha256=receipt['checkpoint_sha256'],source_active_grid_sha256=receipt['grid_archive_sha256'],queries_sha256=sha(qpath),mapping_code_sha256=sha(Path(__file__).with_name('marshawn_semantic_cell_admission.py')),
        resolution=resolution.tolist(),semantic_gate0_cells=int(base.sum()),semantic_pad1_cells=int(gate.sum()),mapped_admitted_cells=int(mapped.sum()),total_LC_cells=int(mapped.size),
        actual_active_before=int(active.sum()),hypothetical_active_intersection=int((active[0]&mapped).sum()),all_semantic_source_cell_centers_retained=True,
        witness_rows=witness_rows,all103_witnesses_retained=all(r['xyz_retained']and r['pixel_ray_at_depth_retained']for r in witness_rows),toe_anchors=toe_rows,
        traces=traces,coordinate_epsilon=cover.epsilon,seconds=time.time()-began,source_EVAL_used_for_gate=False,no_model_loaded=True,GPU_used=False,production_mutated=False,
        mapping_policy='Each whole LC cell is admitted if ANY intersected reviewed128grid cell is admitted; closed-boundary contact and outside-grid portions remain UNKNOWN/admitted. Semantic pad1 is distinct from native occupancy halo.',
        limitations=['Reviewed partial occlusion/unknown policy is an admission hypothesis, not a physical-empty-space certificate.','Existing rejected mass is not a rerender prediction or training-quality result.','Full-interval rejection uses a conservative axis-aligned box cover; mixed intervals are retained.','A future training-time test must change only admission, preserve warmup/halo provenance, and be reviewed separately.','Point5568 material ambiguity remains annotated separately; original103 records unchanged.'])
    write(out/'report.json',result);print(json.dumps({k:result[k]for k in ['resolution','mapped_admitted_cells','total_LC_cells','actual_active_before','hypothetical_active_intersection','all103_witnesses_retained','toe_anchors','seconds']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.campaign,a.source,a.output)
