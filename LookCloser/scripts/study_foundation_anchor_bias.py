"""Real-PatchMatch anchored, spatially held-out stereo disparity-bias canary.

No pose fit, held-out RGB, new neural inference, mesh fusion or video rollout.
Same scalar-offset policy for all pairs, including the weak G/A-H/A pair.
"""
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import map_coordinates, binary_erosion
from PIL import Image, ImageDraw
from joint_temporal_texture import read, sha, atomic_json
from study_confidence_depth_prior import support, unproject
from stereo_anchor_bias import fit_and_validate
from calibrated_stereo_rectification import disparity_to_world

ROOT = Path('/mnt/data/dec5_foundation_anchor_bias')
SOURCES = [Path('/mnt/data/dec5_foundation_hand_stereo/001037'), Path('/mnt/data/dec5_foundation_wrist_stereo/001037')]


def sample(array, uv):
    return map_coordinates(array, uv.T[::-1], order=1, mode='constant', cval=0)


def run():
    import study_forearm_plane_transfer_v3 as source
    source.configure(); start = time.monotonic()
    rows, depths, hashes = source.v2.v1.load_real('001037'); lookup = {r['physical_camera']:i for i,r in enumerate(rows)}
    ROOT.mkdir(exist_ok=False)
    request = dict(frame='001037', source_depth_hashes=hashes,
        scripts={str(Path(__file__).with_name(n).resolve()): sha(Path(__file__).with_name(n)) for n in
                 [Path(__file__).name, 'stereo_anchor_bias.py', 'study_confidence_depth_prior.py',
                  'calibrated_stereo_rectification.py', 'study_forearm_plane_transfer.py',
                  'study_forearm_plane_transfer_v2.py', 'study_forearm_plane_transfer_v3.py']},
        measured_anchor_minimum_other_views=3, measured_tolerance=.001, reprojection_pixels=1.5,
        parallax_degrees=1, anchor_native_pixel_offset=0.,
        sampled_rectified_stride=4, spatial_block=128, excluded_block_margin=8,
        minimum_fold_anchors=100, maximum_disparity_offset=4.,
        maximum_fold_offset_spread=1.5, required_cv_median_factor=.9, maximum_cv_p90_factor=1.05,
        correction='one median disparity offset per stereo pair; same rule, not per-frame manual tuning',
        heldout_rgb_used=False, missing_surface_accuracy_not_tested=True, production_updated=False)
    atomic_json(ROOT / 'request.json', request)
    results = []; corrected_maps = []; dependencies = {}
    for root in SOURCES:
        staged = read(root / 'request.json')
        for pair in staged['pairs']:
            name = Path(pair['directory']).name; folder = Path(pair['directory']); dest = ROOT / name; dest.mkdir()
            calpath = folder / 'calibration.npz'; assert sha(calpath) == pair['hashes']['calibration.npz']
            prediction = root / 'inference' / name / 'prediction.npz'
            assert sha(prediction) == read(prediction.parent / 'complete.json')['prediction_sha256']
            assert sha(folder / 'left.png') == pair['hashes']['left.png']
            dependencies.update({str(p):sha(p) for p in [calpath, prediction, folder / 'left.png', root / 'request.json']})
            cal = np.load(calpath); pred = np.load(prediction)
            k, e = cal['cropped_intrinsic'], cal['rectified_extrinsic']
            left = rows[lookup[pair['left']]]; depth = depths[lookup[pair['left']]]
            y, x = np.mgrid[0:768:4, 0:768:4]; y, x = y.ravel(), x.ravel()
            skin = binary_erosion(cal['left_mask'].astype(bool), iterations=3)[y,x]
            nx = np.rint(1919 - cal['left_map_y'][y,x]).astype(int)
            ny = np.rint(cal['left_map_x'][y,x]).astype(int)
            inside = skin & (nx >= 0) & (nx < 1920) & (ny >= 0) & (ny < 1080)
            native = np.unique(np.column_stack((nx[inside], ny[inside])), axis=0)
            nx, ny = native.T; available = depth[ny,nx] > 0; nx, ny = nx[available], ny[available]
            points = unproject(left, nx, ny, depth[ny,nx])
            votes, free = support(points, left, rows, depths)
            trusted = votes >= 3
            p = points @ e[:3,:3].T + e[:3,3]; uv = p @ k.T; uv = uv[:,:2] / uv[:,2:]
            candidate_domain = pred['consistent'] & pred['valid_source_domain'] & cal['left_mask'].astype(bool)
            available_prediction = sample(candidate_domain.astype(float), uv) > .999
            select = trusted & available_prediction & (p[:,2] > 0)
            selected_uv = uv[select]; predicted = sample(pred['left_disparity'], selected_uv)
            fb = float(k[0,0] * cal['baseline']); offset = float(cal['disparity_offset'])
            expected = fb / p[select,2] + offset
            result = fit_and_validate(selected_uv, predicted, expected, fb, offset)
            result.update(pair=name, left=pair['left'], right=pair['right'], raw_measured_queries=len(points),
                trusted_measured_queries=int(trusted.sum()), matched_anchor_count=int(select.sum()))
            np.savez_compressed(dest / 'anchors.npz', native_xy=np.column_stack((nx,ny)),
                points=points, other_votes=votes, raw_free_votes=free, rectified_uv=uv,
                selected=select, predicted=predicted, expected=expected)
            correction = result['correction'] if result['correction'] is not None else 0.
            yy, xx = np.indices(pred['left_disparity'].shape)
            for mode, delta in [('original',0.), ('offset_diagnostic',correction)]:
                disp = pred['left_disparity'] + delta
                xyz = disparity_to_world(xx,yy,disp,k,e,float(cal['baseline']),offset)
                right = sample(cal['right_mask'].astype(float), np.column_stack((xx.ravel()-pred['left_disparity'].ravel(),yy.ravel()))).reshape(xx.shape) > .999
                valid = candidate_domain & right
                corrected_maps.append(dict(pair=name,mode=mode,xyz=xyz,valid=valid,
                    depth=fb/(disp-offset),K=k,E=e))
            # Visible measured-anchor residuals, not an image-quality heatmap.
            im = Image.new('RGB',(1536,800)); im.paste(Image.open(folder / 'left.png'),(0,32))
            im.paste(Image.open(folder / 'left.png'),(768,32)); draw = ImageDraw.Draw(im)
            draw.text((4,8),name+' / input',fill='white')
            draw.text((772,8),'PM - learned disparity: red positive / blue negative; +/-4 px',fill='white')
            for (u,v), residual in zip(selected_uv, expected-predicted):
                color = (255,int(255*(1-min(abs(residual)/4,1))),0) if residual >= 0 else (0,int(255*(1-min(abs(residual)/4,1))),255)
                draw.ellipse((u+766,v+30,u+770,v+34),fill=color)
            im.save(dest / 'anchor_review.png')
            result['anchors_sha256'] = sha(dest / 'anchors.npz')
            atomic_json(dest / 'result.json',result); results.append(result)
            print(name,result,flush=True)
    comparisons = []
    for a in corrected_maps:
        points = a['xyz'][a['valid']][::4]
        for b in corrected_maps:
            if a['mode'] != b['mode'] or a['pair'] == b['pair']: continue
            p = points @ b['E'][:3,:3].T + b['E'][:3,3]; uv = p @ b['K'].T; uv = uv[:,:2]/uv[:,2:]
            overlap = (p[:,2]>0) & (sample(b['valid'].astype(float),uv)>.999)
            difference = sample(b['depth'],uv)[overlap] - p[overlap,2]
            comparisons.append(dict(mode=a['mode'],source=a['pair'],other=b['pair'],overlap=int(overlap.sum()),
                median=float(np.median(abs(difference))) if len(difference) else None,
                p90=float(np.percentile(abs(difference),90)) if len(difference) else None))
    np.savez_compressed(ROOT / 'point_fields.npz', **{m['pair']+'_'+m['mode']+'_'+key:m[key]
        for m in corrected_maps for key in ['xyz','valid','depth','K','E']})
    atomic_json(ROOT / 'result.json',dict(pairs=results,comparisons=comparisons,dependencies=dependencies,
        request_sha256=sha(ROOT/'request.json'), elapsed_seconds=time.monotonic()-start,
        cross_pair_difference_not_ground_truth_error=True,visual_status='pending',production_updated=False))
    print('finished',time.monotonic()-start,flush=True)


if __name__ == '__main__': run()
