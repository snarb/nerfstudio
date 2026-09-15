"""Audit and freeze a rejected geometry proposal; never publish it as a repair."""
import argparse
from pathlib import Path
import subprocess
import time
import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes
from joint_temporal_texture import read,sha,atomic_json,project,HELD_CAMERAS
from local_silhouette_volume import combine_silhouettes,remove_box_caps
from silhouette_domain_surface import availability_bits,stable_domain_faces
import study_hand_silhouette_volume as base
import study_wide_hand_silhouette as wide

SCRIPTS=['local_silhouette_volume.py','study_hand_silhouette_volume.py','review_hand_silhouette_volume.py',
         'stage_wider_hand_observations.py','silhouette_domain_surface.py','study_wide_hand_silhouette.py',
         'review_wide_hand_silhouette.py','freeze_hand_silhouette_study.py']


def run(check):
    manifest=wide.ROOT/'artifact_manifest.json'
    if check:
        q=read(manifest)
        for p,h in q['files'].items():
            if sha(p)!=h:raise ValueError('Changed artifact '+p)
        print('Verified',len(q['files']),'retained/input hashes',flush=True);return
    if manifest.exists():raise ValueError('Already frozen; use --check')
    aroot=base.ROOT/base.FRAME;broot=wide.EXTRA/base.FRAME
    a=base.verify();b=read(broot/'request.json');request=read(wide.ROOT/'request.json')
    assert sha(aroot/'request.json')==request['original_request_sha256']
    assert sha(broot/'request.json')==request['extra_request_sha256']
    external=[]
    for q,root in [(a,aroot),(b,broot)]:
        assert not q['heldout_used'] and not q['target_camera_used_for_geometry']
        assert len(q['cameras'])==6
        assert not (set(r['camera']['physical_camera'] for r in q['cameras'])&HELD_CAMERAS)
        for p,h in q['dependencies'].items():assert sha(p)==h;external.append(Path(p))
        for n,h in q['scripts'].items():assert sha(Path(__file__).with_name(n))==h
        assert sha(root/'silhouettes.npz')==q['silhouettes_sha256']
    assert sha(Path(wide.__file__))==request['script_sha256']
    assert sha(Path(__file__).with_name('silhouette_domain_surface.py'))==request['domain_helper_sha256']
    data=dict(np.load(aroot/'silhouettes.npz'));data.update(dict(np.load(broot/'silhouettes.npz')))
    replay=[]
    for name,records in [('six',a['cameras']),('twelve',a['cameras']+b['cameras'])]:
        rows=[r['camera'] for r in records];fields=[data[r['physical_camera']+'_field'] for r in rows]
        domains=[data[r['physical_camera']+'_domain'] for r in rows]
        archive=np.load(wide.ROOT/name/'field.npz');field=archive['field'];bits=archive['bits']
        lower=archive['lower'];spacing=float(archive['spacing'])
        selected=np.random.default_rng(17).choice(field.size,2048,replace=False)
        ijk=np.array(np.unravel_index(selected,field.shape)).T
        uv,z=project(lower+ijk*spacing,rows)
        observed,_,_=combine_silhouettes(uv,z,fields,domains,3,0.)
        np.testing.assert_allclose(observed,field.ravel()[selected],rtol=0,atol=1e-5)
        np.testing.assert_array_equal(availability_bits(uv,z,domains),bits.ravel()[selected])
        v,t,_,_=marching_cubes(field,0,spacing=(spacing,)*3);v+=lower
        box=remove_box_caps(v,t,lower,lower+(np.array(field.shape)-1)*spacing,spacing)
        stable=stable_domain_faces(v,t,lower,spacing,bits,3)
        for mode,keep in [('raw',box),('domain_safe',box&stable)]:
            path=wide.ROOT/name/mode/'hull.ply';result=read(path.parent/'result.json')
            assert sha(path)==result['mesh_sha256']
            original=o3d.io.read_triangle_mesh(str(path))
            model=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(t[keep]))
            model.remove_unreferenced_vertices()
            np.testing.assert_array_equal(np.asarray(model.vertices),np.asarray(original.vertices))
            np.testing.assert_array_equal(np.asarray(model.triangles),np.asarray(original.triangles))
            assert len(model.triangles)==result['triangles']
        replay.append(dict(camera_count=len(rows),replayed_field_samples=2048,
                           replayed_raw_faces=int(box.sum()),replayed_domain_faces=int((box&stable).sum())))
    # Matched old-six result is not a different stochastic surface.
    assert sha(aroot/'margin0/hull.ply')==sha(wide.ROOT/'six/raw/hull.ply')
    viewed=[aroot/'masks_native.png',broot/'masks_native.png',
        base.OBS/base.FRAME/'six_train_views_native.png',
        Path('/mnt/data/dec5_wrist_wide_observations/001037/six_train_views_native.png')]
    viewed += [aroot/'geometry_review'/(n+'.png') for n in ['G004_A005_121071','H004_A005_1210M6','elevated_workaround_00099']]
    viewed += [wide.ROOT/'review'/(n+'.png') for n in ['G004_A005_121071','H004_A005_1210M6','E004_C005_1210YM','F004_D005_1210KW','elevated_workaround_00099']]
    atomic_json(wide.ROOT/'visual_review.json',dict(status='fail_as_surface_replacement',reviewer='main_agent',
        inspected={str(p):sha(p) for p in viewed},
        notes='Six-camera envelope inflates badly in extra E/C and F/D views. Twelve-camera intersection constrains gross extent, but fingers remain fused, wrist/hand shell is incomplete and depth is not validated. Removing domain-edge facets removes invented walls, not the need for anatomical/depth evidence.',
        color_mask_limits='Warm held lipstick and narrow room leaks are included; the masks are not certified anatomical segmentation.',
        review_scope='12 listed observation/mask/geometry panels; no RGB prediction or new movie accepted',
        production_updated=False,artifact_free_approval=False))
    atomic_json(wide.ROOT/'audit.json',dict(replay=replay,matched_six_raw_mesh_bytes=True,heldout_used=False,
        production_mesh_unchanged=sha(a['source_mesh'])==a['source_mesh_sha256'],
        no_new_quality_metrics=True,geometry_proposal_rejected=True,
        scope='Replayed sampled volume values and every extracted raw/domain-safe face; not anatomical validity or observed-depth approval.'))
    ps=subprocess.check_output(['ps','-eo','pid,etime,pcpu,rss,args'],text=True)
    running=[p for p in ps.splitlines() if any(('python scripts/'+n) in p for n in SCRIPTS[:-1]) and '/bin/bash' not in p]
    if running:raise ValueError('Study jobs still running: '+str(running))
    atomic_json(wide.ROOT/'terminal_check.json',dict(unix_time=time.time(),study_workers=running,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
        disk=subprocess.check_output(['df','-h','/mnt/data'],text=True),all_launched_sessions_reported_exit=True))
    roots=[base.ROOT,wide.EXTRA,wide.ROOT,Path('/mnt/data/dec5_wrist_wide_observations')]
    files={str(p):sha(p) for root in roots for p in root.rglob('*') if p.is_file() and p!=manifest}
    external += viewed+[Path(__file__).with_name(n) for n in SCRIPTS]
    external += [Path(__file__).parents[1]/'tests/test_local_silhouette_volume.py',
                 Path(__file__).parents[1]/'experiments/dec5_hand_silhouette_volume.md']
    external += list(Path('/mnt/data').glob('dec5_hand_silhouette_*.log'))
    external += list(Path('/mnt/data').glob('dec5_wrist_wide_observations*.log'))
    files.update({str(p):sha(p) for p in external})
    atomic_json(manifest,dict(files=files,status='rejected_geometric_proposal_frozen',production_updated=False))
    print('Replayed four complete face inventories; froze',len(files),'hashes',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');run(p.parse_args().check)
