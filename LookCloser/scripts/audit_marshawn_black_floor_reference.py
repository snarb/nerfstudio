"""CPU-only approximate-floor support table for the archived nine black rays."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from scipy.spatial import ConvexHull,Delaunay,cKDTree


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def hull_distance(point,polygon):
    best=float('inf')
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        t=np.clip((point-a)@(b-a)/max(float((b-a)@(b-a)),1e-20),0,1)
        best=min(best,float(np.linalg.norm(point-(a+t*(b-a)))))
    return best


def main(root):
    folder=root/'native_png_batch1M_sched22k_LOCAL_STAGED/black_KA_audit'
    plane_path=root/'splatfacto_gain4/floor_geometry_diagnosis/manual_floor_fit.npz';points_path=root/'geometry/train_witnesses.npz';tracks_path=root/'geometry/strict_tracks.json'
    f=np.load(plane_path);w=np.load(points_path);tracks=json.loads(tracks_path.read_text());n=f['normal'];d=float(f['offset']);ids=f['point_indices'][f['selected_inliers']];points=w['xyz'][ids];names=w['cameras'].tolist();tree=cKDTree(points);tri=Delaunay(points[:,:2]);hull=points[ConvexHull(points[:,:2]).vertices,:2]
    cold=json.loads((folder/'cold8k/audit.json').read_text());warm=json.loads((folder/'uniform13k/audit.json').read_text());rows=[]
    for i,(a,b)in enumerate(zip(cold['rows'],warm['rows'])):
        if a['pixel']!=b['pixel']or a['origin']!=b['origin']or a['direction']!=b['direction']:raise ValueError('Actual rays differ')
        o=np.array(a['origin']);v=np.array(a['direction']);t=-float(o@n+d)/float(v@n);xyz=o+v*t;inside=bool(tri.find_simplex(xyz[:2])>=0);distances,neighbors=tree.query(xyz,k=3)
        sources=[]
        for distance,j in zip(distances,neighbors):
            idx=int(ids[j]);sources.append(dict(point_index=idx,XYZ=points[j].tolist(),distance3D=float(distance),signed_plane_residual=float(points[j]@n+d),actual_SfM_observer_train_cameras=[names[c]for c in tracks['camera_ids'][idx]],manual_floor_ROI_votes=int(f['manual_floor_view_votes'][idx])))
        row=dict(id=i,pixel=a['pixel'],reference_plane_ray_t=t,reference_plane_XYZ=xyz.tolist(),inside_15point_XY_hull=inside,XY_distance_to_hull=0. if inside else hull_distance(xyz[:2],hull),nearest_vetted_floor_witnesses=sources,
                 vetted_points_within_radius={str(radius):int((np.linalg.norm(points-xyz,axis=1)<=radius).sum())for radius in [.05,.1,.2,.4]},models={})
        for label,data in [('cold8k',a),('uniform13k',b)]:
            p=data['profiles']['uniform32768'];median=next(q for q in p['termination_quantiles']if q['q']==.5);mean_xyz=o+v*p['expected_depth'];q50_xyz=np.array(median['xyz'])
            row['models'][label]=dict(dense_opacity=p['opacity'],conditional_mean_t=p['expected_depth'],conditional_mean_signed_plane_height=float(mean_xyz@n+d),conditional_q50_t=median['t'],conditional_q50_XYZ=median['xyz'],conditional_q50_signed_plane_height=float(q50_xyz@n+d),surface_depth_error_claimed=False)
        rows.append(row)
    result=dict(plane_source=str(plane_path),plane_sha256=sha(plane_path),normal=n.tolist(),offset=d,support_points=len(points),point_indices=ids.tolist(),support_XY_extent=[points[:,:2].min(0).tolist(),points[:,:2].max(0).tolist()],selected_fit_max_abs_residual=float(abs(points@n+d).max()),strict_points_sha256=sha(points_path),actual_tracks_sha256=sha(tracks_path),source_audit_sha256={k:sha(folder/k/'audit.json')for k in ['cold8k','uniform13k']},rows=rows,
        limitations=['The15points areold manuallyreviewed trainSfMfloor-covering references, notindependentdepthGT.','ManualROIvote isclassification support, not3Daccuracycertificate; actualobserverlists doNOTproveallobserversunoccludedfloor.','Supportincludesnear-duplicates;fitresidualdoesNOTquantifyabsolute/extrapolatedplaneuncertainty.','AllrayplaneintersectionsaretestedagainstXYhull andlocalpointdistances; outsidehull impliesextrapolation.','Barefloorandcoveringpanelsneednotshareexactheight; nohardplaneopacity/depthclaim orcarve.','Denseconditionaldepth compareslearnedtermination, notmeasuredphysicalsurface.'],no_training=True,no_plane_fit_or_geometry_edit=True)
    (folder/'floor_reference_support.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Approximate floor reference — diagnostic only','',
           '| Pixel | In hull | Nearest floor witness | Ref t | Cold mean t / q50 height | Safe13k mean t / q50 height |',
           '|---|---|---:|---:|---:|---:|']
    for q in rows:
        a,b=q['models']['cold8k'],q['models']['uniform13k'];lines.append(f'| {q["pixel"]} | {q["inside_15point_XY_hull"]} | {q["nearest_vetted_floor_witnesses"][0]["distance3D"]:.3f} | {q["reference_plane_ray_t"]:.3f} | {a["conditional_mean_t"]:.3f} / {a["conditional_q50_signed_plane_height"]:+.3f} | {b["conditional_mean_t"]:.3f} / {b["conditional_q50_signed_plane_height"]:+.3f} |')
    lines+=['','Distances/heights are prepared-world units, not claimed meters. Negative height is below the extrapolated reference, not proved below true floor.','No independent physical-opacity certificate exists at these pixels from this sparse reference.']
    (folder/'floor_reference_support.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/fsx/tmp/marshawn_001660_hd'));a=p.parse_args();main(a.root)
