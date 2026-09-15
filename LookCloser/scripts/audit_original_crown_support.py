"""Recompute accepted two-frame evidence, then seal read-only diagnosis."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import numpy as np
import open3d as o3d
from joint_temporal_texture import read,sha,atomic_json
from diagnose_original_crown_support import ROOT,MASKS,SOURCE,FRAMES,classify
from study_confidence_depth_prior import load_real
from study_jaw_depth_footprint import train_reference_votes
from study_jaw_repair_transfer import mask_votes


def run():
    ps=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
    live=[p for p in ps if any('python scripts/'+n in p for n in
        ['diagnose_original_crown_support.py','review_original_crown_projections.py']) and '/bin/bash' not in p]
    assert not live,live
    hashes={};records=[]
    for frame in FRAMES:
        folder=ROOT/frame;q=read(folder/'request.json');r=read(folder/'result.json');a=np.load(folder/'evidence.npz')
        assert sha(folder/'request.json')==r['request_sha256']
        for p,h in q['scripts'].items():assert sha(p)==h,p;hashes[p]=h
        for p,h in q['rgb_receipt']['source_rgb_hashes'].items():assert sha(p)==h,p;hashes[p]=h
        for p,h in r['hashes'].items():assert sha(folder/p)==h,p
        assert sha(q['source_mesh'])==q['source_mesh_sha256'];hashes[q['source_mesh']]=q['source_mesh_sha256']
        base=read(SOURCE/frame/'request.json');rows,depths,receipt=load_real(Path(base['depth_root']),frame)
        assert receipt==q['depth_receipt']
        mesh=o3d.io.read_triangle_mesh(q['source_mesh']);v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles)
        selected=t[a['query_triangles']]
        points=np.concatenate([v[selected],v[selected].mean(1)[:,None]],axis=1)
        np.testing.assert_array_equal(points,a['points'])
        votes,refs=train_reference_votes(points.reshape(-1,3),rows,depths)
        np.testing.assert_array_equal(votes.reshape(-1,4),a['depth_votes'])
        np.testing.assert_array_equal(refs.reshape(-1,4),a['depth_references'])
        names=read(MASKS/frame/'cameras.json');masks=np.load(MASKS/frame/'masks.npz')['masks']
        assert sha(MASKS/frame/'masks.npz')==q['refined_masks_sha256']
        s,o=mask_votes(v,selected,rows,masks,names)
        np.testing.assert_array_equal(s,a['refined_mask_support']);np.testing.assert_array_equal(o,a['refined_mask_outside'])
        low,multi,both=classify(votes.reshape(-1,4),o)
        for key,value in [('low_support',low),('multi_mask_outside',multi),('both',both)]:np.testing.assert_array_equal(value,a[key])
        projection=read(folder/'projection_review.json')
        assert len(projection['records'])==3
        for p in projection['records']:
            assert sha(p['panel'])==p['panel_sha256']
            assert p['clear_background_cameras']==sum(x['all_four_clear_background'] for x in p['observations'])
        records.append(dict(frame=frame,query_triangles=len(selected),sample_votes_recomputed=len(points)*4,
            refined_mask_votes_recomputed=True,source_mesh_unchanged=True))
        for p in folder.rglob('*'):
            if p.is_file():hashes[str(p)]=sha(p)
        print(frame,'replayed',len(selected),'triangles',flush=True)
    inspected=[ROOT/f/'native_crown.png' for f in FRAMES]
    inspected += [ROOT/f/f'projection_{i:02d}.png' for f in FRAMES for i in range(3)]
    atomic_json(ROOT/'visual_review.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        inspected_images={str(p):sha(p) for p in inspected},status='evidence_for_targeted_fringe_test_not_deletion_approval',
        notes='Native crown labels and all six six-camera projection panels inspected. Selected weak triangles '
              'often project onto clear background in separated cameras, including several native appearances; '
              'other views overlap hair/curls. The final123 example has five votes at one vertex: preserve this '
              'contradictory positive evidence. Labels alone do not justify deleting all fringe or filling all dark gaps.',
        geometry_changed=False,production_updated=False,quality_approval=False))
    atomic_json(ROOT/'audit.json',dict(records=records,workers_terminal=True,geometry_changed=False,
        excluded_attempts=['001083_failed_index_provenance','001123_failed_index_provenance',
                          '001083_invalid_display_review','001123_invalid_display_review'],
        quality_metrics_computed=False,production_updated=False))
    for n in ['diagnose_original_crown_support.py','review_original_crown_projections.py',Path(__file__).name]:
        p=Path(__file__).resolve().with_name(n);hashes[str(p)]=sha(p)
    for n in ['visual_review.json','audit.json','projection_summary.json']:hashes[str(ROOT/n)]=sha(ROOT/n)
    atomic_json(ROOT/'artifact_manifest.json',dict(hashes=hashes,geometry_changed=False,production_updated=False))
    print('Sealed',len(hashes),'accepted hashes',flush=True)


def check():
    hashes=read(ROOT/'artifact_manifest.json')['hashes']
    for p,h in hashes.items():assert sha(p)==h,p
    print('Rechecked',len(hashes),'accepted hashes',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();check() if a.check else run()
