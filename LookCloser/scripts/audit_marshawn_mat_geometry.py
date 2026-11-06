"""TRAIN material-point surface diagnostics; never edits a density field or gate.

Convex hulls below bound interpolation proposals. They do not certify opaque
coverage, occupied volume, calibrated uncertainty, or permission to delete it.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from audit_marshawn_geometry import projection


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_plane(xyz):
    center = xyz.mean(axis=0)
    _, singular, vh = np.linalg.svd(xyz-center, full_matrices=False)
    normal = vh[-1]
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    return center, normal, vh[:2], singular


def ray_bounded_slab(origin, direction, center, normal, basis, hull, band):
    """Intersect convex interpolation footprint and normal residual band."""
    lower, upper = 0., np.inf
    constraints = [(normal, -normal@center-band), (-normal, normal@center-band)]
    for row in hull.equations:
        n = row[:2]@basis
        constraints.append((n, row[2]-n@center))
    for n, b in constraints:
        value, slope = n@origin+b, n@direction
        if abs(slope) < 1e-12:
            if value > 0:
                return None
        elif slope > 0:
            upper = min(upper, -value/slope)
        else:
            lower = max(lower, -value/slope)
    return [float(lower), float(upper)] if lower <= upper else None


def diagnose(qualification, transforms, box, output):
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads(qualification.read_text())
    tf = json.loads(transforms.read_text())
    cameras = {r['physical_camera']:r for r in tf['frames'] if r['file_path'] in tf['train_filenames']}
    assert len(cameras) == 63 and not(set(cameras)&{'DC','HC','KA'})
    transform = np.asarray(json.loads(box.read_text())['world_to_box'])
    inverse = np.linalg.inv(transform)
    records = {r['source_point_id']:r for r in source['records']}
    point_results, covariances = [], {}
    for pid, row in records.items():
        if not(row['qualified_mat_material'] or row['label']=='black_shoe_sole_material'):
            continue
        xyz = np.asarray(row['xyz_floor'])
        original = inverse[:3,:3]@xyz+inverse[:3,3]
        jacobians = []
        for ob in row['source_graph']['observations']:
            assert ob['camera'] in cameras
            p, *_ = projection(cameras[ob['camera']])
            q = p@np.r_[original,1.]
            j = (p[:2,:3]*q[2]-q[:2,None]*p[2,:3])/q[2]**2
            jacobians.extend(j@inverse[:3,:3])
        jacobians = np.asarray(jacobians)
        cov = .5**2*np.linalg.inv(jacobians.T@jacobians)
        covariances[pid] = cov
        eigen = np.linalg.eigvalsh(cov)
        point_results.append(dict(source_point_id=pid,label=row['label'],xyz_floor=xyz.tolist(),
            covariance_floor_assuming_independent_half_HD_pixel_sigma=cov.tolist(),
            three_sigma_major_axis_world=float(3*np.sqrt(eigen[-1])),
            fit_error_max_HD_px=row['source_graph']['fit_error_max_hd_px'],
            leave_one_original_source_view_out=row['source_graph']['leave_one_source_view_out']))
    groups = {
        'upper_cover_and_upper_rim': [p for p,r in records.items() if r['eligible_upper_surface_proxy']],
        'front_long_rim_and_side': [p for p,r in records.items() if r['qualified_mat_material'] and r['label'] in ['mat_upper_rim','mat_side_panel','mat_lower_rim','mat_side_seam_below_lip'] and r['xyz_floor'][1]<0],
        'near_shoe_short_rim_and_side': [p for p,r in records.items() if r['qualified_mat_material'] and r['label'] in ['mat_upper_rim','mat_side_panel'] and r['xyz_floor'][1]>.1],
    }
    fits, objects = [], {}
    for name, ids in groups.items():
        # Avoid weighting almost-identical physical detections twice.
        kept, duplicates = [], []
        for pid in ids:
            close = [q for q in kept if np.linalg.norm(np.asarray(records[pid]['xyz_floor'])-records[q]['xyz_floor'])<.001]
            if close:
                duplicates.append(dict(excluded=pid,representative=close[0],rule='within .001 world units'))
            else:
                kept.append(pid)
        xyz = np.asarray([records[p]['xyz_floor'] for p in kept])
        center, normal, basis, singular = fit_plane(xyz)
        coordinates = (xyz-center)@basis.T
        hull = ConvexHull(coordinates)
        residual = (xyz-center)@normal
        loo = []
        for i in range(len(xyz)):
            c,n,_,_ = fit_plane(np.delete(xyz,i,axis=0))
            loo.append(float(abs((xyz[i]-c)@n)))
        sigma3 = np.asarray([3*np.sqrt(normal@covariances[p]@normal) for p in kept])
        band = float(max(np.max(np.abs(residual)),max(loo))+max(sigma3))
        fit = dict(name=name,source_point_ids=kept,near_duplicate_exclusions=duplicates,
            center=center.tolist(),normal=normal.tolist(),basis=basis.tolist(),singular_values=singular.tolist(),
            convex_interpolation_hull_floor=(center+coordinates[hull.vertices]@basis).tolist(),
            rms_normal_residual_world=float(np.sqrt(np.mean(residual**2))),max_normal_residual_world=float(np.max(np.abs(residual))),
            max_leave_one_point_out_residual_world=max(loo),diagnostic_half_band_world=band,
            band_rule='max(in-sample absolute residual, leave-one-point-out residual) + largest fixed-calibration 3-sigma normal localization; empirical diagnostic, not coverage guarantee',
            per_point=[dict(source_point_id=p,normal_residual_world=float(e),leave_one_point_out_residual_world=l,localization_3sigma_normal_world=float(s)) for p,e,l,s in zip(kept,residual,loo,sigma3)])
        fits.append(fit);objects[name]=(center,normal,basis,hull,band)
    anchors = {'toe_upper_three_TRAIN_views':[-.11769208960984658,.15127923559030565,.014567836967476211], 'toe_sole_original_SIFT_6862':records[6862]['xyz_floor']}
    rays = []
    for camera in ['DE','EE']:
        pose = transform@np.asarray(cameras[camera]['transform_matrix'])
        origin = pose[:3,3]
        for label, xyz in anchors.items():
            delta = np.asarray(xyz)-origin;distance=np.linalg.norm(delta);direction=delta/distance
            original=inverse[:3,:3]@xyz+inverse[:3,3];p,*_=projection(cameras[camera]);q=p@np.r_[original,1.]
            intersections=[]
            for name,args in objects.items():
                interval=ray_bounded_slab(origin,direction,*args)
                intersections.append(dict(surface=name,interval_world=interval,wholly_before_shoe=bool(interval and interval[1]<distance),entry_to_shoe_distance=None if not interval else float(distance-interval[0])))
            rays.append(dict(camera=camera,shoe_anchor=label,xyz_floor=xyz,source_projection_COLMAP_HD=(q[:2]/q[2]).tolist(),origin_floor=origin.tolist(),direction_floor=direction.tolist(),shoe_distance_world=float(distance),bounded_proxy_intersections=intersections))
    result=dict(kind='TRAIN_MAT_BOUNDED_GEOMETRY_PROPOSAL_NOT_A_CARVING_RULE',qualification_sha256=digest(qualification),transforms_sha256=digest(transforms),box_sha256=digest(box),
        qualified_mat_points=sum(r['qualified_mat_material'] for r in source['records']),point_localization=point_results,surface_proxies=fits,shoe_occlusion_ray_crosschecks=rays,
        limitations=['No gate, field, checkpoint, or accepted box modified.','TRAIN original feature observations and manually reviewed material identity; no EVAL input or model density used.',
        'Sparse points and convex hull do not certify continuous opaque material, holes, back edge, underside, or occupied volume.',
        'Fixed-camera half-HD-pixel linearized uncertainty excludes rig calibration bias, correspondence bias, and surface deformation; doubling pixel sigma doubles localization axes.',
        'Surface residual bands are empirical diagnostics and do not bound all unmeasured mat wrinkles.',
        'A plane/hull ray crossing explains a possible occlusion only; it does not alone authorize clear space or deletion.','Near-shoe protection covers measured points, not entire feet.'])
    (output/'bounded_geometry_proposal.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(15,6));ax=fig.add_subplot(121,projection='3d');plan=fig.add_subplot(122)
    colors={'mat_upper_rim':'tab:blue','mat_top_material':'tab:cyan','mat_side_panel':'tab:orange','mat_handle':'tab:purple','mat_lower_rim':'tab:brown','mat_side_seam_below_lip':'tab:red','black_shoe_sole_material':'black'}
    for label,color in colors.items():
        rr=[r for r in records.values() if r['label']==label]
        if not rr:continue
        xyz=np.asarray([r['xyz_floor'] for r in rr]);ax.scatter(*xyz.T,c=color,label=label,s=25);plan.scatter(xyz[:,0],xyz[:,1],c=color,s=25)
        for r in rr:plan.annotate(str(r['source_point_id']),r['xyz_floor'][:2],fontsize=6)
    for f in fits:
        hull=np.asarray(f['convex_interpolation_hull_floor']);hull=np.r_[hull,hull[:1]];ax.plot(*hull.T,alpha=.5);plan.plot(hull[:,0],hull[:,1],alpha=.5)
    ax.set(xlabel='floor x',ylabel='floor y',zlabel='floor z',title='Measured points; hull outlines are interpolation proposals');ax.set_box_aspect([1,1,.4]);ax.legend(fontsize=6)
    plan.set(xlabel='floor x',ylabel='floor y',title='TRAIN material points, toe protection, sparse footprint');plan.set_aspect('equal');plan.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'qualified_points_and_proxy_hulls.png',dpi=150);plt.close(fig)
    from PIL import Image, ImageDraw
    for camera in ['DE','EE']:
        source_path=transforms.parent/cameras[camera]['file_path']
        im=Image.open(source_path).convert('RGB');draw=ImageDraw.Draw(im)
        p,*_=projection(cameras[camera])
        def project(x):
            original=np.asarray(x)@inverse[:3,:3].T+inverse[:3,3]
            q=np.c_[original,np.ones(len(original))]@p.T
            assert (q[:,2]>0).all()
            return q[:,:2]/q[:,2,None]-.5  # COLMAP center to array coordinates
        extents=[]
        for f,color in zip(fits,[(0,255,255),(255,160,0),(190,100,255)]):
            uv=project(f['convex_interpolation_hull_floor']);extents.extend(uv)
            draw.line([tuple(x) for x in np.r_[uv,uv[:1]]],fill=color,width=2)
        for label,xyz in anchors.items():
            uv=project([xyz])[0];extents.append(uv);x,y=uv
            draw.line((x-6,y,x+6,y),fill=(255,40,40),width=2);draw.line((x,y-6,x,y+6),fill=(255,40,40),width=2)
            draw.text((x+8,y-10),'upper' if 'upper' in label else 'sole',fill=(255,80,80))
        uv=np.asarray(extents);left,top=np.maximum(0,np.floor(uv.min(axis=0)-70)).astype(int);right,bottom=np.minimum(im.size,np.ceil(uv.max(axis=0)+70)).astype(int)
        crop=im.crop((left,top,right,bottom));draw=ImageDraw.Draw(crop);draw.rectangle((0,0,crop.width,25),fill=(0,0,0));draw.text((5,6),camera+' TRAIN: sparse proxy outlines; red = occluded shoe anchors',fill='white');crop.save(output/(camera+'_source_proxy_diagnostic.png'))
    print(json.dumps(dict(surfaces=[{k:f[k] for k in ['name','rms_normal_residual_world','max_normal_residual_world','max_leave_one_point_out_residual_world','diagnostic_half_band_world']} for f in fits],rays=rays)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['qualification','transforms','box','output']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();diagnose(**vars(args))
