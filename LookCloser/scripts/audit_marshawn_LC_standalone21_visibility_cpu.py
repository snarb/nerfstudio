"""Offline native contribution/floor-height diagnostics; never changes sigma."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

C=Path('/fsx/tmp/marshawn_001660_hd/tuning_20261106')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mass_row(query_id,camera,xyz,w,starts,ends,sigma):
    w=w.astype(np.float64);A=float(w.sum());z=xyz[:,2];cuts=[-.05,-.03,-.02,0.]
    below=[float(w[z<cut].sum())for cut in cuts]
    visible=w>0;dominant=int(np.argmax(w))if len(w)else None
    serial=np.empty(len(w),np.float64);tau_before=0.
    for i,(start,end,s)in enumerate(zip(starts,ends,sigma)):
        tau=float(s)*(float(end)-float(start))
        serial[i]=math.exp(-tau_before)*-math.expm1(-tau);tau_before+=tau
    error=float(abs(serial-w).max())if len(w)else 0.
    if error>2e-5:raise ValueError('Independent serial source weights differ')
    return dict(query_id=query_id,camera=camera,native_alpha=A,samples=len(w),
        visible_samples=int(visible.sum()),serial_FP64_weight_max_error=error,
        below_cut_native_mass=dict(zip(map(str,cuts),below)),
        below_cut_fraction_of_native_A=dict(zip(map(str,cuts),[v/A if A else None for v in below])),
        retained_above_cut_fraction=dict(zip(map(str,cuts),[1-v/A if A else None for v in below])),
        visible_z_min=float(z[visible].min())if visible.any()else None,
        visible_z_max=float(z[visible].max())if visible.any()else None,
        dominant_native_weight=float(w[dominant])if dominant is not None else 0.,
        dominant_XYZ=xyz[dominant].tolist()if dominant is not None else None)


def main():
    p=Path(sys.argv[1]);out=p/'CPU_visibility_v1';out.mkdir(exist_ok=True)
    completion=json.loads((p/'capture/complete.json').read_text());manifest=json.loads((p/'query_manifest.json').read_text())
    plane_file=Path(__file__).resolve().parents[1]/'configs/aabb_review/marshawn_1660_1900_floor.json'
    plane=json.loads(plane_file.read_text());fit=plane['floor_fit']
    row3=np.r_[fit['floor_plane_normal'],fit['floor_plane_offset']]
    assert np.array_equal(row3,np.array(plane['world_to_box'])[2])
    residual=np.array(fit['floor_points_prepared_world'])@row3[:3]+row3[3]
    assert abs(float(abs(residual).max())-fit['max_fit_residual'])<1e-14
    records=[];source_hashes={str(p/'capture/complete.json'):sha(p/'capture/complete.json'),
        str(p/'query_manifest.json'):sha(p/'query_manifest.json'),str(plane_file):sha(plane_file)}
    by_query={q['query_id']:dict(camera=r['camera'],**q)for r in completion['records']for q in r['queries']}
    for r in completion['records']:
        file=p/'capture'/(r['camera']+'.npz');assert sha(file)==r['archive_sha256'];source_hashes[str(file)]=sha(file)
        with np.load(file,allow_pickle=False)as z:
            for i,query_id in enumerate(z['query_id']):
                choose=z['ray_indices']==i;fields=[z[k][choose]for k in ['xyz','weight','starts','ends','sigma']]
                record=mass_row(str(query_id),r['camera'],*fields);record.update(
                    namespace='new_standalone_capture',pixel_xy=z['pixel_xy'][i].tolist(),
                    semantic_target=float(z['semantic_target'][i]),current_fit_weight=float(z['label_weight'][i]),
                    category=str(z['category'][i]),cap_hit=bool(z['cap_hit'][i]),
                    pixel_area=float(z['actual_pixel_area'][i,0]),depth_mass50=by_query[str(query_id)]['depth_native_mass50'],
                    depth_mass95=by_query[str(query_id)]['depth_native_mass95'])
                records.append(record)
    original=[]
    for q in manifest['existing_five_provenance']:
        file=C/'frozen_LC_semantic_capture_plan_v4/capture'/(q['camera']+'.npz');source_hashes[str(file)]=sha(file)
        with np.load(file,allow_pickle=False)as z:
            choose=z['ray_indices']==q['camera_local_index'];fields=[z[k][choose]for k in ['xyz','weight','starts','ends','sigma']]
            row=mass_row('original_cached_'+str(q['global_query_index']),q['camera'],*fields)
            row.update(namespace='original_frozen_128ray_packet_cache',pixel_xy=q['pixel_xy'],
                semantic_target=q['original_target'],current_fit_weight=q['original_label_weight'])
            original.append(row)
    low_file=p/'capture/seven_low_cell_densities.npz';source_hashes[str(low_file)]=sha(low_file)
    with np.load(low_file)as z:
        low=[dict(cell=cell.tolist(),xyz_declared=xyz.tolist(),sigma=float(s))for cell,xyz,s in zip(z['cell'],z['xyz_declared_float64'],z['sigma'])]
    result=dict(passed=True,source_bindings=source_hashes,new21=records,original5=original,seven_cell_point_densities=low,
        floor_contract=dict(z_is_already_floor_coordinate=True,no_second_world_to_box=True,
            original_world_to_floor_row3=row3.tolist(),fitted_plane_floor_z=0,
            max_fit_residual=fit['max_fit_residual'],local_mat_warp_bound=None,
            units=fit['units']),
        interpretation='Native source contribution mass below each fixed z cut; no hard carve, new semantic fit, sigma edit or opticalalphaGT. Individual query fractions do not certify complete actor geometry or local floor shape.',
        original5_context_equivalence_claimed=False,all_new21_fit_weights_remain0=True,CUDA_started=False)
    (out/'visibility_and_floor_mass.json').write_text(json.dumps(result,indent=2)+'\n')
    with(out/'floor_mass_all26.csv').open('w')as f:
        fields=['namespace','query_id','camera','native_alpha','samples','visible_z_min','visible_z_max',
            'below_m005_fraction','below_m003_fraction','below_m002_fraction','below_0_fraction']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in records+original:
            q={k:row[k]for k in fields[:7]};q.update(zip(fields[7:],row['below_cut_fraction_of_native_A'].values()));writer.writerow(q)
    print(json.dumps(dict(output=str(out),rays=len(records),original_rays=len(original),point_density_queries=len(low),
        serial_FP64_weight_max_error=max(q['serial_FP64_weight_max_error']for q in records+original),CUDA_started=False)))


if __name__=='__main__':main()
