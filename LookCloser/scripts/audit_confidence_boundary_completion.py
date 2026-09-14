"""Independent assembly invariants plus fresh 62-view dual-lattice color guard."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_dilation
from joint_temporal_texture import read,sha,atomic_json
from study_confidence_depth_prior import project_integer
from annotation_mask_domain import semantic_faces
from audit_color_qualified_forearm import run as fresh_guard
import study_forearm_plane_transfer_v3 as prior

def run(root,frame):
    prior.configure();v1=prior.v2.v1;folder=root/frame;r=read(folder/'request.json');g=read(folder/'geometry_result.json')
    for name,digest in r['scripts'].items():
        if sha(Path(__file__).with_name(name))!=digest:raise ValueError('Changed assembly script')
    if r['parameters'].get('known_annotation_margin')!=3:raise ValueError('Wrong annotation-domain control')
    e=np.load(folder/'evidence.npz');old=np.load(prior.OUT/frame/'plane/evidence.npz');data=np.load(prior.OUT/frame/'diagnostic.npz')
    md=data[v1.NAMES[0]+'_mesh'];trusted=data[v1.NAMES[0]+'_trusted']
    expected=binary_dilation(old['accepted'])&((md>0)|old['accepted']);pins=expected&~old['accepted']&trusted
    if not np.array_equal(e['domain'],expected) or not np.array_equal(e['accepted'],old['accepted']):raise ValueError('Changed proposal domain')
    if not np.array_equal(e['measured_pins'],pins) or not np.array_equal(e['depth'][pins],md[pins]):raise ValueError('Measured boundary moved')
    source=o3d.io.read_triangle_mesh(r['source_mesh']);mesh=o3d.io.read_triangle_mesh(str(folder/'guarded.ply'))
    v,t=np.asarray(mesh.vertices),np.asarray(mesh.triangles);nv,nt=len(source.vertices),len(source.triangles)
    if not np.array_equal(v[:nv],np.asarray(source.vertices)) or not np.array_equal(t[:nt],np.asarray(source.triangles)):raise ValueError('Original prefix changed')
    rows,_,_=v1.cameras(frame);ref=next(row for row in rows if row['physical_camera']==v1.NAMES[0]);uv,z=project_integer(ref,v[nv:]);y,x=np.nonzero(expected)
    if not np.allclose(uv,np.column_stack([x,y]),atol=1e-3) or not np.allclose(z,e['depth'][y,x],atol=1e-6):raise ValueError('Saved geometry differs from solved rays')
    kept,sem=semantic_faces(v,t[nt:],rows,v1.masks(frame),axis_extent=True)
    if len(kept)!=len(t)-nt:raise ValueError('Final geometry violates known semantic domain')
    fresh_guard(root,frame)
    atomic_json(folder/'independent_audit.json',dict(status='measured_pins_domain_original_prefix_and_124_rays_pass',
        measured_pins=int(pins.sum()),final_added_triangles=len(kept),semantic=sem,
        geometry_result_sha256=sha(folder/'geometry_result.json'),fresh_ray_audit_sha256=sha(root/'fresh_audit'/(frame+'.json')),
        script_sha256=sha(__file__),artifact_free=False,geometry_is_inferred_not_measured=True))
    print(frame,'independent assembly audit pass',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frame',required=True);p.add_argument('--root',type=Path,default=Path('/mnt/data/dec5_forearm_measured_boundary_domain'));a=p.parse_args();run(a.root,a.frame)
