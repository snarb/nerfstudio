"""Seal the inspected visibility diagnostic, rejected global control and fallback."""
from pathlib import Path
import numpy as np
import open3d as o3d
from study_multiview_face_prior import read,save,sha
from recover_supported_front_surface import ROOT,CULLED,PARENT,FRAME
from diagnose_gap_texture_admission import ROOT as DIAGNOSTIC
from review_measured_free_surface import VIEWS


def main():
    out=ROOT/FRAME/'visual_review.json';assert not out.exists();bindings={};viewed=[]
    def verify(p,h):
        assert sha(p)==h,str(p)
        bindings[str(p)]=h
    for folder in [DIAGNOSTIC,DIAGNOSTIC/'witnesses',CULLED/FRAME/'review',ROOT/FRAME/'review']:
        r=read(folder/'result.json')
        for p,h in r['input_hashes'].items():verify(p,h)
        for p,h in r.get('outputs',{}).items():verify(folder/p,h)
        for p,h in r.get('images',{}).items():verify(folder/p,h)
        if 'evidence_sha256' in r:verify(folder/'evidence.npz',r['evidence_sha256'])
        verify(folder/'result.json',sha(folder/'result.json'))
    viewed+=sorted((DIAGNOSTIC/'witnesses').glob('*.png'))
    for view in VIEWS:
        root=CULLED/FRAME/view;r=read(root/'culling_audit.json')
        verify(root/'request.json',r['request_sha256'])
        verify(root/'frames'/FRAME/'complete.json',r['frame_complete_sha256'])
        verify(root/'target_retained_faces.npy',r['retained_faces_sha256'])
        viewed += [CULLED/FRAME/'review'/view/name for name in ['head_native.png','lipstick_native.png']]
        folder=ROOT/FRAME/view;q=read(folder/'result.json')
        for p,h in q['hashes'].items():verify(folder/p,h)
        verify(Path(__file__).with_name('recover_supported_front_surface.py'),q['script_sha256'])
    reviewed=read(ROOT/FRAME/'review/result.json')
    for r in reviewed['records']:viewed += [Path(c['path']) for c in r['components']]
    assert len(viewed)==17
    e=np.load(DIAGNOSTIC/'evidence.npz');r=read(PARENT/'carved'/FRAME/'rgb/moving/frames'/FRAME/'result.json')
    mesh=o3d.io.read_triangle_mesh(r['mesh_path']);assert sha(r['mesh_path'])==r['mesh_sha256']
    v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles)[e['face_ids']];p=e['points']
    n=np.cross(v[t[:,1]]-v[t[:,0]],v[t[:,2]]-v[t[:,0]]);n/=np.linalg.norm(n,axis=1)[:,None]
    direction=np.array(r['camera']['transform_matrix'])[:3,3]-p;direction/=np.linalg.norm(direction,axis=1)[:,None]
    incidence=np.sum(n*direction,axis=1)
    assert (incidence[:4]<0).all() and incidence[4]>0
    save(out,dict(reviewer='root LLM actual native comparison inspection',
        viewed_images={str(p):sha(p) for p in viewed},checked_bindings=bindings,checked_count=len(bindings),
        target_incidence_of_diagnostic_points=incidence.tolist(),
        global_culling_verdict='fail: removes the hand black cluster but worsens existing crown holes and produces new missing contour pixels',
        guarded_moving_verdict='verified narrow repair: four-pixel hand-side cluster replaced by train texture of a supported deeper front surface; no other pixel changed',
        guarded_HC_verdict='no change',guarded_KB_verdict='five isolated pixels recovered; existing ragged hair geometry remains, no claim of crown repair',
        direct_visibility_is_mesh_evidence_not_independent_physical_truth=True,
        no_new_black_pixels_in_guarded_variant=True,original_colored_pixels_unchanged=True,
        geometry_file_unchanged=True,ray_visibility_changed=True,artifact_free=False,
        video_unchanged=True,production_promoted=False,temporal_transfer_not_tested=True,
        script_sha256=sha(__file__)))
    print('verified',len(bindings),'bindings;17 images reviewed;global rejected;guarded narrow recovery',flush=True)


if __name__=='__main__':main()
