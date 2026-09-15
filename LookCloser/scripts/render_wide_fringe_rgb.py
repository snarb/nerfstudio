"""Two wide-angle RGB transfer canaries for the sealed fringe replacement."""
from copy import deepcopy
from pathlib import Path
import argparse
import numpy as np
from joint_temporal_texture import read,sha,atomic_json
from review_jaw_repair_transfer import verified_image,panel

ROOT=Path('/mnt/data/dec5_wide_fringe_rgb/left_high_arc')
PARENT=Path('/mnt/data/dec5_large_motion_choices_v3/left_high_arc')
MESH=Path('/mnt/data/dec5_weak_fringe_replacement')
FRAMES=['001083','001123']


def render():
    import render_smooth_temporal_mesh_video as engine
    from run_view_consistent_dynamic_video import install
    implementation=install();engine.torch.set_num_threads(2)
    request=deepcopy(engine.verify_request(PARENT))
    assert implementation==request['source_quality_implementation_sha256']
    request['inventory']=[r for r in request['inventory'] if r['frame_id'] in FRAMES]
    request['ordered_frame_ids']=FRAMES
    request['source_rows']=[r for r in request['source_rows'] if Path(r['source_dataset']).name in FRAMES]
    bindings={}
    for entry in request['inventory']:
        frame=entry['frame_id'];source=read(MESH/frame/'request.json')
        assert entry['mesh_sha256']==source['source_mesh_sha256']
        folder=MESH/frame/'replace';receipt=read(folder/'result.json')
        assert receipt['revealed_shell_guard_passed'] and sha(folder/'mesh.ply')==receipt['hashes']['mesh.ply']
        bindings[frame]=dict(request_sha256=sha(MESH/frame/'request.json'),result_sha256=sha(folder/'result.json'))
        entry.update(mesh=str(folder/'mesh.ply'),mesh_sha256=sha(folder/'mesh.ply'))
    request.update(partial_diagnostic_only=True,full_video_candidate=False,artifact_free_approval=False,
        weak_fringe_bindings=bindings,inferred_shell_not_measured=True,geometry_changed=True,
        texture_source_masks_unchanged=True,matched_parent_sha256=sha(PARENT/'request.json'))
    request['script_hashes'][Path(__file__).name]=sha(__file__)
    (ROOT/'frames').mkdir(parents=True,exist_ok=True)
    if (ROOT/'request.json').exists(): assert read(ROOT/'request.json')==request
    else: atomic_json(ROOT/'request.json',request)
    engine.render(ROOT,FRAMES)


def review():
    request=read(ROOT/'request.json');parent=read(PARENT/'request.json');records=[]
    assert sha(PARENT/'request.json')==request['matched_parent_sha256']
    for key in ('profiles_sha256','exposure_sha256','calibration_sha256'): assert request[key]==parent[key]
    for frame in FRAMES:
        old,oldr=verified_image(PARENT,frame);new,newr=verified_image(ROOT,frame)
        for key in ('camera','source_cameras','fixed_exposure'):assert oldr[key]==newr[key]
        ds=[np.rot90(np.load(p/'frames'/frame/'target_depth.npz')['depth']) for p in (PARENT,ROOT)]
        for d in ds: assert d.shape==(1920,1080) and np.isfinite(d).all()
        before,after=ds;lost=(before>0)&(after==0);gained=(before==0)&(after>0)
        # Same box for both images, derived only from the baseline's top extent.
        yy,xx=np.nonzero(before>0);top=int(yy.min());xs=xx[yy<top+450]
        center=int(np.median(xs));left=max(0,min(1080-760,center-380))
        boxes={'crown':(left,max(0,top-10),left+760,min(1920,top+360)),
               'head':(left,max(0,top-10),left+760,min(1920,top+850))}
        for name,box in boxes.items():panel(ROOT/'review'/frame/(name+'.png'),[old,new],['production','fringe replacement'],box)
        # Cross-check against previously completed CPU casts at this exact pose.
        screen=Path('/mnt/data/dec5_wide_fringe_geometry')/frame/'left_high_arc'
        prior=read(screen/'result.json');assert prior['camera']==oldr['camera']
        assert sha(screen/'depths.npz')==prior['output_hashes']['depths.npz']
        expected=np.load(screen/'depths.npz')
        np.testing.assert_allclose(before,expected['production'],rtol=0,atol=1e-6)
        np.testing.assert_allclose(after,expected['replace'],rtol=0,atol=1e-6)
        records.append(dict(frame=frame,lost_depth=int(lost.sum()),new_depth=int(gained.sum()),
            changed_rgb=int(np.any(old!=new,axis=2).sum()),
            black_introduced=int(((old.max(2)>0)&(new.max(2)==0)).sum()),
            black_removed=int(((old.max(2)==0)&(new.max(2)>0)).sum()),boxes=boxes,
            render_hashes=dict(production=sha(PARENT/'frames'/frame/'frame.png'),candidate=sha(ROOT/'frames'/frame/'frame.png')),
            panel_hashes={name:sha(ROOT/'review'/frame/(name+'.png')) for name in boxes},
            independent_wide_depths_match=True))
    atomic_json(ROOT/'review/result.json',dict(records=records,request_sha256=sha(ROOT/'request.json'),
        counts_not_quality_metrics=True,visual_status='pending',full_video_approval=False))
    print(records,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('render','review'));a=p.parse_args()
    (render if a.action=='render' else review)()
