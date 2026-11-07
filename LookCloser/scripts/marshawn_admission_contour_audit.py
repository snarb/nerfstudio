"""Read-only dense nominal-contour ray audit of a source-derived admission grid.

Exact DDA traverses half-open voxels; this is a necessary image-ray check, not
an assertion about hidden surface depth or physical empty-space correctness.
No dataset, model, source mask or production gate is changed.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
from scipy.ndimage import binary_erosion


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_helpers(data):
    import sys
    code=data.parents[2]/'code'
    scripts=code/'LookCloser/scripts'
    if not scripts.exists():raise ValueError('Frame-native frozen code tree required')
    sys.path[:0]=[str(code),str(scripts)]
    return scripts


def ray_hits(grid, bounds, origin, directions):
    """Vectorized half-open cell DDA, without sampling gaps between cells."""
    directions = np.asarray(directions, np.float64)
    origin = np.broadcast_to(np.asarray(origin, np.float64), directions.shape)
    bounds = np.asarray(bounds, np.float64)
    shape = np.asarray(grid.shape)
    pitch = (bounds[1] - bounds[0]) / shape
    parallel = np.abs(directions) < 1e-14
    outside = (parallel & ((origin < bounds[0]) | (origin > bounds[1]))).any(1)
    inv = np.divide(1., directions, out=np.zeros_like(directions), where=~parallel)
    a = (bounds[0] - origin) * inv
    b = (bounds[1] - origin) * inv
    entry = np.where(parallel, -np.inf, np.minimum(a, b)).max(1)
    exit = np.where(parallel, np.inf, np.maximum(a, b)).min(1)
    entry = np.maximum(entry, 0.)
    valid = ~outside & np.isfinite(entry) & np.isfinite(exit) & (exit > entry)
    hit = np.zeros(len(directions), bool)
    ids = np.flatnonzero(valid)
    if not len(ids):
        return hit, valid
    d, o, near, far = directions[ids], origin[ids], entry[ids], exit[ids]
    # A point infinitesimally after entry identifies the first half-open cell.
    point = o + np.nextafter(near, far)[:, None] * d
    cell = np.clip(np.floor((point - bounds[0]) / pitch).astype(np.int64), 0, shape - 1)
    step = np.sign(d).astype(np.int64)
    edge = bounds[0] + (cell + (step > 0)) * pitch
    tmax = np.divide(edge - o, d, out=np.full_like(d, np.inf), where=np.abs(d) > 1e-14)
    delta = np.divide(pitch, np.abs(d), out=np.full_like(d, np.inf), where=np.abs(d) > 1e-14)
    active = np.ones(len(ids), bool)
    for _ in range(int(shape.sum()) + 4):
        local = np.flatnonzero(active)
        if not len(local):
            break
        retained = grid[tuple(cell[local].T)]
        hit[ids[local[retained]]] = True
        active[local[retained]] = False
        local = local[~retained]
        if not len(local):
            continue
        next_t = tmax[local].min(1)
        end = next_t >= far[local]
        active[local[end]] = False
        local = local[~end]
        next_t = next_t[~end]
        if not len(local):
            continue
        advance = tmax[local] <= next_t[:, None] + 1e-13
        cell[local] += step[local] * advance
        tmax[local] += np.where(advance, delta[local], 0.)
        escaped = ((cell[local] < 0) | (cell[local] >= shape)).any(1)
        active[local[escaped]] = False
    if active.any():
        raise RuntimeError('DDA traversal did not terminate')
    return hit, valid


def audit(data, admission, output, cameras=None, chunk=4096, learned=None):
    output.mkdir(parents=True, exist_ok=True)
    ready = json.loads((data / 'ready.json').read_text())
    transforms = json.loads((data / 'transforms.json').read_text())
    rows = {r['physical_camera']: r for r in transforms['frames']}
    with np.load(admission / 'semantic_grid.npz', allow_pickle=False) as z:
        grids = {'source_base': z['semantic_gate0'].copy(), 'source_pad1': z['semantic_gate_pad1'].copy()}
        bounds = z['aabb'].copy()
    if any(g.dtype != bool for g in grids.values()) or not np.array_equal(bounds, np.asarray(ready['aabb'])):
        raise ValueError('Grid/data bounds or dtype mismatch')
    helper_root=source_helpers(data)
    from marshawn_semantic_cell_admission import WholeCellAdmission
    grids['source_mapped'] = WholeCellAdmission(grids['source_pad1'], bounds).map_target_grid((128,)*3, bounds)
    grid_bounds = {k: bounds for k in grids}
    if learned:
        with np.load(learned, allow_pickle=False) as z:
            lb = z['cascade_aabbs'].reshape(-1,2,3)
            if len(lb) != 1 or not np.allclose(lb[0], bounds, rtol=0, atol=2e-7):
                raise ValueError('Learned runtime grid is not the same bounded scene')
            for key in ['core', 'active', 'admission']:
                a = z[key]
                if a.shape != (1,128,128,128) or a.dtype != bool:
                    raise ValueError('Unexpected learned occupancy format')
                grids['runtime_'+key] = a[0].copy()
                grid_bounds['runtime_'+key] = lb[0]
    before = dict(ready=sha(data / 'ready.json'), transforms=sha(data / 'transforms.json'),
                  grid=sha(admission / 'semantic_grid.npz'))
    if learned:
        before['runtime_snapshot'] = sha(learned)
    records = []
    start = time.time()
    for name, row in rows.items():
        if cameras and name not in cameras:
            continue
        path = data / 'semantics' / (name + '.npz')
        semantic_sha = sha(path)
        pose = np.asarray(row['transform_matrix'])
        with np.load(path, allow_pickle=False) as z:
            for actor in ['black', 'grey']:
                mask = z[actor]
                boundary = mask & ~binary_erosion(mask, border_value=0)
                y, x = np.where(boundary)
                for kind, grid in grids.items():
                    failed, outside = [], 0
                    for offset in range(0, len(x), chunk):
                        xx, yy = x[offset:offset + chunk], y[offset:offset + chunk]
                        local = np.stack([(xx + .5 - row['cx']) / row['fl_x'],
                                          -(yy + .5 - row['cy']) / row['fl_y'], -np.ones(len(xx))], 1)
                        directions = local @ pose[:3, :3].T
                        hit, valid = ray_hits(grid, grid_bounds[kind], pose[:3, 3], directions)
                        outside += int((~valid).sum())
                        failed.extend(np.stack([xx[~hit], yy[~hit]], 1).tolist())
                    target = output / (name + '_' + actor + '_' + kind + '_failed.npz')
                    np.savez_compressed(target, pixel_HD_xy=np.asarray(failed, np.int32).reshape(-1, 2))
                    record = dict(camera=name, actor=actor, grid_kind=kind, boundary_rays=len(x), failed=len(failed),
                                  outside_box=outside, failed_pixels=str(target), failed_pixels_sha256=sha(target),
                                  semantics_sha256=semantic_sha)
                    if failed:
                        from PIL import Image, ImageDraw
                        image_path = data / 'images' / ('train_'+name+'.png')
                        image = Image.open(image_path).convert('RGB'); draw = ImageDraw.Draw(image)
                        for u,v in failed: draw.point((u,v),fill=(255,0,0))
                        image.thumbnail((1200,700)); overlay = target.with_suffix('.png'); image.save(overlay)
                        record.update(failed_overlay=str(overlay),failed_overlay_sha256=sha(overlay),RGB_sha256=sha(image_path))
                    records.append(record)
                    print(json.dumps(record), flush=True)
        if sha(path) != semantic_sha:
            raise ValueError('Source semantics changed during read-only audit')
    after = dict(ready=sha(data / 'ready.json'), transforms=sha(data / 'transforms.json'),grid=sha(admission / 'semantic_grid.npz'))
    if learned: after['runtime_snapshot'] = sha(learned)
    if before != after:
        raise ValueError('Input changed during audit')
    result = dict(kind='READ_ONLY_DENSE_NOMINAL_CONTOUR_DDA_AUDIT_V1', frame=ready['frame'],
                  data=str(data), admission=str(admission), input_sha256=before, code_sha256=sha(__file__),
                  whole_cell_helper_sha256=sha(helper_root/'marshawn_semantic_cell_admission.py'),
                  records=records, counts_by_grid={k:dict(rays=sum(r['boundary_rays']for r in records if r['grid_kind']==k),failed=sum(r['failed']for r in records if r['grid_kind']==k))for k in grids},seconds=time.time()-start,
                  no_inputs_or_model_mutated=True,
                  limitations=['Half-open voxel centerline DDA; exact grazing-boundary ownership is conventional.',
                               'A retained cell on an actor ray does not certify the actual hidden surface voxel.',
                               'Nominal mask contours do not include all potentially missed native hair.'])
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def native_hair_audit(data, admission, output, selection, candidate_root, chunk=4096):
    """Audit ROOT-selected native SAM3 tile contours using explicit lens inversion."""
    import cv2
    helper_root=source_helpers(data)
    from marshawn_actor_coordinates import raw_camera, raw_array_indices, RIG
    from marshawn_semantic_cell_admission import WholeCellAdmission
    output.mkdir(parents=True,exist_ok=True)
    reviewed=json.loads(selection.read_text());ready=json.loads((data/'ready.json').read_text())
    if reviewed['frame']!=ready['frame']:raise ValueError('Native candidate frame differs')
    rows={r['physical_camera']:r for r in json.loads((data/'transforms.json').read_text())['frames']}
    with np.load(admission/'semantic_grid.npz',allow_pickle=False)as z:
        bounds=z['aabb'].copy();grids={'source_base':z['semantic_gate0'].copy(),'source_pad1':z['semantic_gate_pad1'].copy()}
    if not np.array_equal(bounds,np.asarray(ready['aabb'])):raise ValueError('Wrong native audit bounds')
    grids['source_mapped']=WholeCellAdmission(grids['source_pad1'],bounds).map_target_grid((128,)*3,bounds)
    records=[]
    for name,pick in reviewed['selection'].items():
        row=rows[name];spec_path=candidate_root/name/'spec.json';spec=json.loads(spec_path.read_text())
        if spec['frame']!=ready['frame']or spec['camera']!=name:raise ValueError('Native ROI source identity differs')
        path=candidate_root/'remote_results'/name/'candidates'/(pick['prompt']+'.npz')
        archive_sha=sha(path)
        receipt_path=path.parent/'receipt.json';receipt=json.loads(receipt_path.read_text())
        original=candidate_root.parents[1]/'sources'/str(ready['frame'])/name
        photograph=original/'raw_native.png';original_receipt=original/'receipt.json'
        selected_record=next(r for r in receipt['records']if r['prompt'].replace(' ','_')==pick['prompt'])
        source_record=json.loads(original_receipt.read_text())
        data_record=next(r for r in ready['records']if r['camera']==name)
        if (receipt['source_spec_sha256']!=sha(spec_path)or receipt['source_receipt_sha256']!=sha(original_receipt)
            or receipt['checkpoint_sha256']!=spec['checkpoint_sha256']or selected_record['archive_sha256']!=archive_sha
            or spec['raw_native_image_sha256']!=sha(photograph)or source_record['image_sha256']!=data_record['rgb_sha256']):
            raise ValueError('Native candidate/NN/source/RGB bindings differ')
        with np.load(path,allow_pickle=False)as z:
            mask=z['masks'][pick['mask_id']].astype(bool);crop=z['crop_xyxy'].astype(int)
        if crop.tolist()!=spec['crop_xyxy']or mask.shape!=(crop[3]-crop[1],crop[2]-crop[0]):raise ValueError('Native tile origin/extent differs')
        yy,xx=np.where(mask&~binary_erosion(mask,border_value=0));raw_xy=np.stack([xx+crop[0],yy+crop[1]],1).astype(np.float64)
        K,d=raw_camera(name);HD=np.array([[row['fl_x'],0,row['cx']-.5],[0,row['fl_y'],row['cy']-.5],[0,0,1.]])
        uv=cv2.undistortPointsIter(raw_xy.reshape(-1,1,2),K,d,None,HD,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-12)).reshape(-1,2)
        back=raw_array_indices(uv+.5,row);residual=float(np.max(np.linalg.norm(back-raw_xy,axis=1)))if len(uv)else 0.
        if residual>1e-4:raise ValueError('Native pixel-center distortion roundtrip failed')
        pose=np.asarray(row['transform_matrix']);records_by_grid=[]
        for kind,grid in grids.items():
            failed=[]
            for offset in range(0,len(uv),chunk):
                part=uv[offset:offset+chunk];local=np.stack([(part[:,0]+.5-row['cx'])/row['fl_x'],-(part[:,1]+.5-row['cy'])/row['fl_y'],-np.ones(len(part))],1)
                hit,_=ray_hits(grid,bounds,pose[:3,3],local@pose[:3,:3].T);failed.extend((np.flatnonzero(~hit)+offset).tolist())
            failed=np.asarray(failed,np.int64);target=output/(name+'_'+kind+'_native_failed.npz')
            np.savez_compressed(target,raw_pixel_xy=raw_xy[failed],HD_array_xy=uv[failed])
            record=dict(grid_kind=kind,rays=len(uv),failed=len(failed),failed_pixels=str(target),failed_pixels_sha256=sha(target))
            if kind=='source_mapped'and len(failed):
                from PIL import Image,ImageDraw
                photo=Image.open(photograph).convert('RGB').crop(tuple(crop));shade=np.zeros((mask.shape[0],mask.shape[1],4),np.uint8);shade[mask]=[0,255,255,55]
                photo=Image.alpha_composite(photo.convert('RGBA'),Image.fromarray(shade));draw=ImageDraw.Draw(photo)
                for u,v in raw_xy[failed]:draw.point((int(u-crop[0]),int(v-crop[1])),fill=(255,0,0))
                overlay=output/(name+'_source_mapped_native_failures.png');photo.save(overlay)
                record.update(native_failure_overlay=str(overlay),native_failure_overlay_sha256=sha(overlay))
            records_by_grid.append(record)
        record=dict(camera=name,actor='black',ROOT_role='qualified_positive_hair'if pick['positive_addition']else'geometry_UNKNOWN_seed_only',
                    candidate_path=str(path),candidate_archive_sha256=archive_sha,candidate_index=pick['mask_id'],candidate_prompt=pick['prompt'],
                    native_crop_xyxy=crop.tolist(),candidate_spec_sha256=sha(spec_path),source_receipt_sha256=spec['source_receipt_sha256'],
                    NN_receipt_sha256=sha(receipt_path),NN_worker_sha256=receipt['worker_code_sha256'],
                    raw_native_image_sha256=spec['raw_native_image_sha256'],sam3_checkpoint_sha256=spec['checkpoint_sha256'],
                    native_pixel_center_roundtrip_max_error=residual,grids=records_by_grid)
        records.append(record);print(json.dumps(record),flush=True)
        if sha(path)!=archive_sha:raise ValueError('Native candidate changed during audit')
    result=dict(kind='READ_ONLY_ROOT_SELECTED_NATIVE_HAIR_CONTOUR_DDA_V1',frame=ready['frame'],records=records,
                source_ready_sha256=sha(data/'ready.json'),transforms_sha256=sha(data/'transforms.json'),grid_sha256=sha(admission/'semantic_grid.npz'),
                ROOT_selection_sha256=sha(selection),code_sha256=sha(__file__),calibration_sha256={n:sha(RIG/n)for n in ['cameras.txt','images.txt']},
                coordinate_helper_sha256=sha(helper_root/'marshawn_actor_coordinates.py'),whole_cell_helper_sha256=sha(helper_root/'marshawn_semantic_cell_admission.py'),
                coordinate_contract='RAW array pixel center -> converged OPENCV undistortion -> HD array center +.5 -> DATA OpenGL c2w; exact forward polynomial roundtrip<=1e-4 native px',
                no_inputs_or_model_mutated=True,limitations=['Native NN proposal contours are not certified anatomical hair truth.','AD low-confidence candidate is audited only as a geometry UNKNOWN seed.'])
    (output/'native_report.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--admission', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cameras', nargs='+')
    p.add_argument('--learned', type=Path)
    p.add_argument('--native-selection', type=Path)
    p.add_argument('--candidate-root', type=Path)
    a = p.parse_args()
    if a.native_selection:
        if not a.candidate_root:p.error('--candidate-root required with --native-selection')
        native_hair_audit(a.data,a.admission,a.output,a.native_selection,a.candidate_root)
    else:
        audit(a.data, a.admission, a.output, a.cameras, learned=a.learned)
