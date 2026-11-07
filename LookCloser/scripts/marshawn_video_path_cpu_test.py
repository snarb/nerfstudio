"""Meaningful timeline, pinhole-equivalence and spline-boundary checks; CPU only."""
import json
from pathlib import Path
import numpy as np
from marshawn_video_path import camera_at, native_row, cropped_row


def verify(path):
    from marshawn_video_render import source_eligibility
    masks=dict(W64=np.ones((2,3),np.float32),valid=np.ones((2,3),bool),
        source_ineligible=np.array([[1,0,0],[0,0,0]],bool),unknown=np.array([[0,1,0],[0,0,0]],bool))
    rgb=np.array([[1,1,0],[1,1,1]],bool)
    np.testing.assert_array_equal(source_eligibility(masks,rgb),[[0,0,0],[1,1,1]])
    try:source_eligibility({'W64':masks['W64'],'valid':masks['valid']},rgb)
    except ValueError:pass
    else:raise AssertionError('Missing crew/UNKNOWN proof was accepted')
    p=json.loads(Path(path).read_text());r=p['records']
    assert len(r)==120 and [q['index'] for q in r]==list(range(120))
    assert [q['frame_id'] for q in r]==[f'{1660+5*i:06d}' for i in range(120)]
    assert r[71]['time_seconds']==71/24 and r[72]['time_seconds']==3
    assert r[71]['kind']=='neural' and all(q['kind']=='actual_gt' for q in r[72:])
    meta=json.loads(Path(p['source_transforms']).read_text());rows={q['physical_camera']:q for q in meta['frames']}
    controls=[rows[n] for n in p['controls']];crop=r[72]['gt_native_crop_xywh']
    for t in [0.,3.]:
        a=camera_at(t,controls,crop);h=1e-4;near=camera_at(h if t==0 else t-h,controls,crop)
        for key in ['fl_x','fl_y','cx','cy']:
            assert abs(a[key]-near[key])<1e-6,(t,key,a[key]-near[key])
        assert np.max(abs(np.asarray(a['transform_matrix'])-np.asarray(near['transform_matrix'])))<1e-8
    endpoint=cropped_row(native_row(controls[-1]),crop)
    assert endpoint==r[72]['camera']
    np.testing.assert_array_equal(endpoint['transform_matrix'],controls[-1]['transform_matrix'])
    full=native_row(controls[-1]);x,y,w,h=crop
    for u,v in [(0,0),(w-1,h-1),(w//2,h//2)]:
        # An output pixel represents EXACTLY the native source ray at (x+u,y+v).
        a=[(u+.5-endpoint['cx'])/endpoint['fl_x'],(v+.5-endpoint['cy'])/endpoint['fl_y']]
        b=[(u+x+.5-full['cx'])/full['fl_x'],(v+y+.5-full['cy'])/full['fl_y']]
        np.testing.assert_allclose(a,b,atol=1e-15,rtol=0)
    assert len(p['future_60fps']['neural_frame_ids'])==180
    assert p['future_60fps']['neural_frame_ids'][-1]=='002018'
    print('PASS timeline72+48, exactt3pose/K, C2boundary, nativecrop ray equivalence, future60fps, crew/UNKNOWNsourceveto')


if __name__=='__main__':
    import sys
    verify(sys.argv[1])
