"""Audit frozen pose probes and prepare native head panels (not video approval)."""
from pathlib import Path
import numpy as np
from PIL import Image
from joint_temporal_texture import read, sha, atomic_json, cameras
from probe_central_train_camera_workaround import ROOT, PARENT, FRAME, COLUMNS
from review_jaw_repair_transfer import verified_image, panel

INTRINSICS = ('fl_x','fl_y','cx','cy','w','h','k1','k2','p1','p2','camera_model')


def validate_camera(camera, native, moving, kind):
    """A foreign-intrinsics target must not inherit a native pixel mask ID."""
    if kind not in ('native','flight_intrinsics'):
        raise ValueError('Unknown intrinsic control')
    np.testing.assert_array_equal(camera['transform_matrix'], native['transform_matrix'])
    if camera['physical_camera'] == native['physical_camera']:
        raise ValueError('Native target mask could be indexed in the wrong pixel gauge')
    if camera['train_pose_physical_camera'] != native['physical_camera']:
        raise ValueError('Wrong physical pose identity')
    expected = native if kind == 'native' else moving
    for key in INTRINSICS:
        if camera.get(key) != expected.get(key):
            raise ValueError('Wrong target intrinsic: '+key)
    if 'file_path' in camera:
        raise ValueError('Target must not carry a source RGB path')


def run():
    request=read(ROOT/'probe_request.json')
    assert sha(PARENT/'request.json')==request['parent_request_sha256']
    assert sha(Path(__file__).with_name('probe_central_train_camera_workaround.py'))==request['script_sha256']
    assert request['probe_only'] and len(request['views'])==15
    views=[r['view'] for r in request['views']];assert len(set(views))==15
    rows,_,_=cameras(FRAME);lookup={r['physical_camera']:r for r in rows};assert len(lookup)==62
    parent=read(PARENT/'request.json');entry=next(r for r in parent['inventory'] if r['frame_id']==FRAME)
    assert sha(entry['mesh'])==entry['mesh_sha256']==request['mesh_sha256']
    out=ROOT/'head_review';out.mkdir(exist_ok=False);records=[]
    expected={'moving'}
    for c in COLUMNS:
        physical=next(n for n in lookup if n.startswith(c+'004_C005_'))
        expected.update(kind+'_'+physical for kind in ['native','flight_intrinsics'])
    assert set(views)==expected
    for record in request['views']:
        view=record['view'];location=ROOT/view;q=read(location/'request.json')
        assert sha(location/'request.json')==record['request_sha256']
        assert len(q['inventory'])==1 and q['inventory'][0]['frame_id']==FRAME
        e=q['inventory'][0];assert e['mesh']==entry['mesh'] and e['mesh_sha256']==entry['mesh_sha256']
        assert e['camera']==record['camera']
        if view=='moving': assert e['camera']==entry['camera']
        else:
            kind='native' if view.startswith('native_') else 'flight_intrinsics'
            validate_camera(e['camera'],lookup[record['physical_camera']],entry['camera'],kind)
        image,result=verified_image(location,FRAME)
        assert image.shape==(1920,1080,3)
        images=[image];labels=[view]
        if view.startswith('native_'):
            images.insert(0,np.array(Image.open(ROOT/'gt'/(record['physical_camera']+'.png'))))
            labels.insert(0,'real train GT')
        path=out/(view+'.png');panel(path,images,labels,(160,450,1000,1250))
        records.append(dict(view=view,panel=str(path),sha256=sha(path),visual_status='pending'))
    atomic_json(out/'audit.json',dict(frame=FRAME,verified_views=15,exact_pose_and_intrinsics_verified=True,
        unchanged_mesh_verified=True,not_a_dynamic_video=True,video_changed=False,
        request_metadata_correction=None if request['not_a_dynamic_video'] else 'probe_request.not_a_dynamic_video=False is a metadata typo; this is a single-time diagnostic, never a dynamic video.',
        no_quality_metrics=True,records=records))
    print('Verified 15 same-time unchanged-mesh views; prepared head panels',flush=True)


if __name__=='__main__':run()
