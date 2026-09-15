"""Independent residual/invariant and retained-file audit of prior conformance."""
from pathlib import Path
import argparse
import numpy as np
from build_train_hair_semantics import read,sha,write
from conform_mhr_measured_surface import ROOT,PARENT,ARMS


def verify():
    import open3d as o3d
    from diffusion_mesh_repair import scene_for
    from triangulate_face_prior import quantiles
    protocol=read(ROOT/'protocol.json')
    assert protocol['script_sha256']==sha(Path(__file__).with_name('conform_mhr_measured_surface.py'))
    for path,digest in protocol['input_hashes'].items():assert sha(path)==digest,path
    assert sha(protocol['original_mesh'])==protocol['original_mesh_sha256']
    assert protocol['recipe']['arms']==ARMS
    initial=np.load(ROOT/'initial.npz');obs=np.load(ROOT/'anchors.npz')
    base=np.load(PARENT/'head20_neck6/fit.npz')['vertices'];tri=initial['triangles']
    subtri=tri[(initial['neutral'][tri,1]>140).all(1)]
    active=initial['neutral'][:,1]>135;validation=obs['validation'][obs['camera']]
    assert int(obs['validation'].sum())==8 and int(validation.sum())==1600
    for arm in ARMS:
        result=read(ROOT/arm/'result.json');fit=np.load(ROOT/arm/'fit.npz')
        assert result['protocol_sha256']==sha(ROOT/'protocol.json')
        assert result['fit_sha256']==sha(ROOT/arm/'fit.npz') and result['mesh_sha256']==sha(ROOT/arm/'prior_only.ply')
        v=fit['vertices'];np.testing.assert_array_equal(v[~active],base[~active])
        np.testing.assert_array_equal(fit['displacement'],v-base)
        mesh=o3d.io.read_triangle_mesh(str(ROOT/arm/'prior_only.ply'))
        np.testing.assert_array_equal(np.asarray(mesh.vertices),v)
        np.testing.assert_array_equal(np.asarray(mesh.triangles),tri)
        scene=scene_for(v,subtri)
        nearest=scene.compute_closest_points(o3d.core.Tensor(obs['points'].astype(np.float32)))['points'].numpy()
        delta=nearest-obs['points'];plane=np.sum(delta*obs['normals'],axis=1);distance=np.linalg.norm(delta,axis=1)
        np.testing.assert_array_equal(plane,fit['plane']);np.testing.assert_array_equal(distance,fit['distance'])
        for split,take in [('train',~validation),('validation',validation)]:
            for name,group in [('face',~obs['neck']),('neck_candidate_group',obs['neck'])]:
                stats=result['stats'][split+'_'+name]
                assert stats['point_plane']==quantiles(abs(plane[take&group]))
                assert stats['surface_distance']==quantiles(distance[take&group])
        old=np.cross(base[tri[:,1]]-base[tri[:,0]],base[tri[:,2]]-base[tri[:,0]])
        new=np.cross(v[tri[:,1]]-v[tri[:,0]],v[tri[:,2]]-v[tri[:,0]])
        reversed_normals=np.sum(old*new,axis=1)<=0
        np.testing.assert_array_equal(reversed_normals,fit['normal_reversed_triangles'])
        assert int(reversed_normals.sum())==result['normal_reversed_triangles']
        assert all(h['lsmr_istop']==[2,2,2] for h in result['history'])
    review=read(ROOT/'review_request.json')
    assert review['script_sha256']==sha(Path(__file__).with_name('review_mhr_measured_conformance.py'))
    assert review['protocol_sha256']==sha(ROOT/'protocol.json') and review['fit_summary_sha256']==sha(ROOT/'fit_summary.json')
    for name,digest in review['reviewer_hashes'].items():assert sha(Path(__file__).with_name(name))==digest
    for sub,name in [('review_smooth025_smooth100_smooth400','manifest.json'),('probe_smooth025_smooth100_smooth400','result.json')]:
        for item in read(ROOT/sub/name)['files']:assert sha(item['path'])==item['sha256']
    independent=Path('/mnt/data/dec5_mhr_conformance_independent_review')
    evidence=read(independent/'result.json')
    assert evidence['source_protocol_sha256']==sha(ROOT/'protocol.json')
    for arm,digest in evidence['source_fit_hashes'].items():assert sha(ROOT/arm/'fit.npz')==digest
    assert evidence['validation_rows_removed_before_replay']==1600 and evidence['train_only_replay_max_vertex_difference']==0
    seal=read(independent/'seal.json')
    assert seal['status']=='passed'
    assert seal['report_sha256']==sha(Path(__file__).resolve().parents[1]/'experiments/dec5_mhr_conformance_independent_review.md')


def main(action):
    verify()
    if action=='seal':
        images=sorted((ROOT/'review_smooth025_smooth100_smooth400').glob('*.png'))+[ROOT/'probe_smooth025_smooth100_smooth400/requested_hole_clay_native.png']
        assert len(images)==7
        write(ROOT/'visual_review.json',dict(reviewer='main LLM',inspected={str(p):sha(p) for p in images},
            status='reject_whole_prior_local_patch_feasibility_improved',
            findings='Under-jaw/neck better follows measured contour; whole priors retain artificial eye/nose/mouth folds and coarse neck triangles. Strongest smoothing visually less folded. No texture, local addition, measured free-space or temporal acceptance.',
            original_mesh_changed=False,production_promoted=False))
        bindings={str(p):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
        independent=Path('/mnt/data/dec5_mhr_conformance_independent_review')
        bindings.update({str(p):sha(p) for p in independent.rglob('*') if p.is_file()})
        report=Path(__file__).resolve().parents[1]/'experiments/dec5_mhr_conformance_independent_review.md'
        bindings[str(report)]=sha(report)
        for name in [Path(__file__).name,'conform_mhr_measured_surface.py','review_mhr_measured_conformance.py']:
            p=Path(__file__).with_name(name).resolve();bindings[str(p)]=sha(p)
        for relative in ['tests/test_mhr_measured_conformance.py','experiments/dec5_mhr_measured_conformance.md']:
            p=Path(__file__).resolve().parents[1]/relative;bindings[str(p)]=sha(p)
        p=Path('/mnt/data/dec5_mhr_measured_conformance_tests.log');bindings[str(p)]=sha(p)
        write(ROOT/'artifact_manifest.json',dict(bindings=bindings,status='prior_conformance_control_complete_not_geometry_repair'))
        print('sealed',len(bindings),'bindings',flush=True)
    else:
        for path,digest in read(ROOT/'artifact_manifest.json')['bindings'].items():assert sha(path)==digest,path
        print('conformance residual/invariant audit passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['seal','check'])
    main(parser.parse_args().action)
