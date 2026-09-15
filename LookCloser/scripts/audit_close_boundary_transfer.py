"""Verify transferred semantic admission and matched RGB without approving quality."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json,cameras
from transfer_close_boundary_completion import ROOT,SOURCE,MOVIE,FRAMES
from study_jaw_repair_transfer import mask_votes
from review_jaw_repair_transfer import verified_image


def audit():
    records=[];hashes={}
    ps=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True)
    names=['transfer_close_boundary_completion.py','render_close_boundary_transfer.py']
    live=[line for line in ps.splitlines() if any('python scripts/'+n in line for n in names) and '/bin/bash' not in line]
    assert not live,'Workers must be terminal before sealing'
    movie=read(MOVIE/'request.json')
    for frame in FRAMES:
        root=ROOT/frame;reqpath=ROOT/(frame+'_controller_request.json');q=read(reqpath)
        for p,digest in q['scripts'].items():
            assert sha(p)==digest,p
            hashes[p]=digest
        base=read(SOURCE/frame/'request.json');assert sha(SOURCE/frame/'request.json')==q['source_request_sha256']
        assert 'mask_override' not in base
        receipt=base['depth_receipt'];depthcal=read(receipt['transforms'])
        mapping={r['physical_camera']:r for r in depthcal['frames']}
        assert len(receipt['depth_sha256'])==62
        for name,digest in receipt['depth_sha256'].items():
            path=Path(receipt['dense'])/'stereo/depth_maps'/(mapping[name]['file_path']+'.geometric.bin')
            assert sha(path)==digest,path
            hashes[str(path)]=digest
        hashes[receipt['transforms']]=sha(receipt['transforms'])
        entry=next(r for r in movie['inventory'] if r['frame_id']==frame)
        maskroot=Path(entry['source_masks']['root']);names=read(maskroot/'cameras.json')
        assert sha(maskroot/'masks.npz')==base['source_mask_sha256']==entry['source_masks']['masks_sha256']
        assert sha(maskroot/'cameras.json')==base['mask_names_sha256']
        masks=np.load(maskroot/'masks.npz')['masks'];rows,_,_=cameras(frame)
        local=o3d.io.read_triangle_mesh(str(root/'local_raw.ply'));v=np.asarray(local.vertices)
        proposals=np.load(root/'proposal_evidence.npz')['proposals'];a=np.load(root/'admission/samples.npz')
        support,outside=mask_votes(v,proposals,rows,masks,names)
        np.testing.assert_array_equal(support,a['mask_support']);np.testing.assert_array_equal(outside,a['mask_outside'])
        np.testing.assert_array_equal(np.flatnonzero((support>=2)&(outside==0)),a['semantic_ids'])
        geometry=root/'interpolated'/frame;g=read(geometry/'result.json');ga=read(geometry/'audit.json')
        assert ga['mesh_sha256']==g['hashes']['mesh.ply']==sha(geometry/'mesh.ply')
        assert len(ga['native_ray_checks'])==124 and all(r['trusted_free_pixels']==0 for r in ga['native_ray_checks'])
        original=o3d.io.read_triangle_mesh(base['source_mesh']);final=o3d.io.read_triangle_mesh(str(geometry/'mesh.ply'))
        np.testing.assert_array_equal(np.asarray(final.vertices)[:len(original.vertices)],np.asarray(original.vertices))
        np.testing.assert_array_equal(np.asarray(final.triangles)[:len(original.triangles)],np.asarray(original.triangles))
        results=[]
        for view in ['moving','F004_E005_1210FP']:
            roots=[MOVIE if view=='moving' else ROOT/'rgb'/frame/view/'baseline',ROOT/'rgb'/frame/view/'completion']
            rs=[]
            for rgbroot in roots:
                im,r=verified_image(rgbroot,frame);rs.append(r)
                assert im.shape==(1920,1080,3) and np.isfinite(im).all()
                hashes[str(rgbroot/'request.json')]=sha(rgbroot/'request.json')
                complete=read(rgbroot/'frames'/frame/'complete.json')
                for name,h in complete['hashes'].items():
                    hashes[str(rgbroot/'frames'/frame/name)]=h
            for key in ['camera','source_cameras','fixed_exposure']:
                assert rs[0][key]==rs[1][key]
            assert rs[0]['mesh_sha256']==base['source_mesh_sha256']
            assert rs[1]['mesh_sha256']==ga['mesh_sha256']
            results.append(dict(view=view,matched_sources_camera_exposure=True))
        records.append(dict(frame=frame,semantic_votes_recomputed=True,original_prefix_exact=True,
            native_guard_checks=124,added_triangles=g['added'],native_guard_clear=True,rgb=results,
            mesh_components=ga['components'],nonmanifold_edges=ga['nonmanifold_edges']))
        for p in [SOURCE/frame/'request.json',Path(base['source_mesh']),maskroot/'masks.npz',maskroot/'cameras.json']:
            hashes[str(p)]=sha(p)
    images=[ROOT/'review'/frame/(view+'_'+part+'.png') for frame in FRAMES
        for view in ['moving','F004_E005_1210FP'] for part in ['head','jaw']]
    atomic_json(ROOT/'visual_review.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        inspected_images={str(p):sha(p) for p in images},
        status='not_promoted_no_meaningful_visible_improvement',
        notes='Both-time moving and real F/E head/jaw comparisons inspected. Crown notches/fringe and tiny under-chin edge defects remain. Three RGB pairs are identical; five existing-hit pixels change at 001123 moving, some become brighter. No additional missing-depth pixel is filled in these views. This does not refute the separate late-frame crack fix, but does not establish a general head repair.',
        full_video_approved=False,production_updated=False,whole_frame_quality_pass=False,
        quality_metrics_computed=False))
    atomic_json(ROOT/'audit.json',dict(utc=datetime.now(timezone.utc).isoformat(),records=records,
        visual_status='reviewed_not_promoted',quality_metrics_computed=False,production_updated=False,
        live_study_workers=live,study_workers_terminal=True))
    for p in ROOT.rglob('*'):
        if p.is_file() and p.name!='artifact_manifest.json':
            hashes[str(p)]=sha(p)
    for n in [Path(__file__).name,'render_close_boundary_transfer.py','transfer_close_boundary_completion.py']:
        p=Path(__file__).resolve().with_name(n);hashes[str(p)]=sha(p)
    for p in Path('/mnt/data').glob('dec5_close_boundary_transfer*.log'):
        # Audit's own log must not be part of its self-changing receipt.
        if p.name.endswith('_audit.log'):
            continue
        hashes[str(p)]=sha(p)
    atomic_json(ROOT/'artifact_manifest.json',dict(hashes=hashes,production_updated=False,visual_approval=False))
    print('Audited',records,'hashes',len(hashes),flush=True)


def check():
    hashes=read(ROOT/'artifact_manifest.json')['hashes']
    for p,h in hashes.items():
        assert sha(p)==h,p
    print('Rechecked',len(hashes),'hashes',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    check() if a.check else audit()
