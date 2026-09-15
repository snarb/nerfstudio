"""Read-only separating-axis witnesses for two specified stored actor outputs."""
from pathlib import Path
from decimal import Decimal,localcontext
import argparse
import numpy as np
from guard_mhr_anatomical_correction import all_pairs
from study_multiview_face_prior import save,sha


def axis_gap(points):
    p=np.asarray(points,np.longdouble);p=p-p.mean(0);a,b=p[:3],p[3:]
    ea=np.roll(a,-1,axis=0)-a;eb=np.roll(b,-1,axis=0)-b
    na=np.cross(ea[0],ea[1]);nb=np.cross(eb[0],eb[1])
    axes=np.r_[na[None,:],nb[None,:],np.cross(ea[:,None,:],eb[None,:,:]).reshape(-1,3),
        np.cross(na,ea),np.cross(nb,eb),np.cross(na,eb),np.cross(nb,ea)]
    length=np.sqrt((axes*axes).sum(1));axes=axes[length>0]/length[length>0,None];axes=np.r_[axes,-axes]
    gaps=(b@axes.T).min(0)-(a@axes.T).max(0);i=gaps.argmax()
    return np.asarray(axes[i],float),float(gaps[i])


def decimal_gap(points,axis):
    with localcontext() as context:
        context.prec=80
        projected=[sum((Decimal(float(x))*Decimal(float(y)) for x,y in zip(p,axis)),Decimal(0)) for p in points]
        return str(min(projected[3:])-max(projected[:3])),[str(x) for x in projected]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();root=args.output
    assert not root.exists();root.mkdir();base=Path('/mnt/data/dec5_mhr_certified_conic_correction')
    paths=[base/'fit.npz',base/'review_v2/silhouette_topology.npz',base/'review_v2/baseline_topology.npz',
        Path('/mnt/data/dec5_mhr_correction_study/last_proposal_diagnosis.npz')]
    fit,topology,warm,proposal=[np.load(path) for path in paths]
    pairs=set(map(tuple,topology['intersections']))-set(map(tuple,warm['intersections']));assert len(pairs)==11
    cases=[('certified_final',fit['vertices'],sorted(pairs)),('anatomical_last_proposal',proposal['proposed'],proposal['new_pairs'])]
    triangles=fit['triangles'];records=[];localtri=np.array([[0,1,2],[3,4,5]])
    for name,vertices,pairs in cases:
        for pair in pairs:
            ids=triangles[np.asarray(pair)].ravel();points=vertices[ids];assert len(set(ids))==6
            rows=[]
            for quantized in [False,True]:
                value=points.astype(np.float32).astype(float) if quantized else points
                axis,gap=axis_gap(value);exact,proj=decimal_gap(value,axis)
                centered=value.astype(np.longdouble)-value.astype(np.longdouble).mean(0)
                centered_gap=float((centered[3:]@axis).min()-(centered[:3]@axis).max())
                cross=np.cross(value[localtri[:,1]]-value[localtri[:,0]],value[localtri[:,2]]-value[localtri[:,0]])
                areas=np.linalg.norm(cross,axis=1);edge=np.linalg.norm(value[localtri[:,1]]-value[localtri[:,0]],axis=1)
                record=dict(float32_quantized=quantized,axis=axis.tolist(),longdouble_max_gap=gap,decimal_axis_gap=exact,
                    centered_axis_gap=centered_gap,world_projections_decimal=proj,cross_norms=areas.tolist(),
                    cross_norm_divided_edge_squared=(areas/edge**2).tolist(),
                    open3d_pair=bool(all_pairs(value,localtri)),coordinate_quantization_max=float(abs(value-points).max()))
                rows.append(record)
            records.append(dict(case=name,pair=list(map(int,pair)),vertex_ids=ids.tolist(),checks=rows))
            np.savez_compressed(root/f'{name}_{pair[0]}_{pair[1]}.npz',points=points,vertex_ids=ids,
                float64_axis=np.array(rows[0]['axis']),float32_axis=np.array(rows[1]['axis']))
    save(root/'result.json',dict(records=records,source_hashes={str(p):sha(p) for p in paths},
        script_sha256=sha(__file__),pair_inventory_from_retained_topology=True,independent_pair_geometry_checks=True,
        actor_fit_launched=False,guard_changed=False,outputs={p.name:sha(p) for p in root.iterdir() if p.is_file()}))
    for case in cases:
        subset=[r for r in records if r['case']==case[0]]
        print(case[0],len(subset),'float64 gap range',min(float(r['checks'][0]['decimal_axis_gap']) for r in subset),
            max(float(r['checks'][0]['decimal_axis_gap']) for r in subset),
            'float32 positive',sum(float(r['checks'][1]['decimal_axis_gap'])>0 for r in subset),flush=True)


if __name__=='__main__':main()
