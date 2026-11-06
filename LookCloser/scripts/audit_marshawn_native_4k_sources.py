"""CPU-only source/calibration feasibility audit; no training dataset mutation."""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def ray(frame,uv):
    q=np.c_[(uv[:,0]-frame['cx'])/frame['fl_x'],-(uv[:,1]-frame['cy'])/frame['fl_y'],-np.ones(len(uv))]
    q=q@np.asarray(frame['transform_matrix'])[:3,:3].T
    return q/np.linalg.norm(q,axis=1)[:,None]


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);root=a.source.parent.parent
    old=root/'color_variants/radial_k4_joint_scalar_DIAGNOSTIC/sdr_png'
    original=json.loads((old/'transforms.json').read_text());high=json.loads((a.source/'transforms.json').read_text());meta=json.loads((a.source/'source_metadata.json').read_text());complete=json.loads((a.source/'complete.json').read_text())
    assert complete['replays_exact'] and complete['eval_nine_files_unchanged']
    native={r['physical_camera']:r for r in original['frames']};high_rows={r['physical_camera']:r for r in high['frames']};sources={r['camera']:r for r in meta['records']};assert set(native)==set(high_rows) and len(native)==66
    world_to_floor=np.asarray(meta['world_to_box']);floor_to_world=np.linalg.inv(world_to_floor);rows=[];proposed=copy.deepcopy(high)
    # Keep original native camera matrices exactly after verifying the rigid relation.
    # Absolute image paths point at immutable sources; this document is not released.
    for camera in sorted(native):
        oldrow=native[camera];newrow=high_rows[camera];train=oldrow['file_path'] in original['train_filenames'];scale=2 if train else 1
        converted=floor_to_world@np.asarray(newrow['transform_matrix']);oldpose=np.asarray(oldrow['transform_matrix']);pose_error=float(np.max(np.abs(converted-oldpose)));assert pose_error<1e-12
        assert all(newrow[k]==oldrow[k]*scale for k in ['w','h','fl_x','fl_y','cx','cy'])
        assert all(newrow.get(k,0)==oldrow.get(k,0)==0 for k in ['k1','k2','k3','k4','p1','p2'])
        rng=np.random.default_rng(20261106);uv=rng.uniform([.5,.5],[oldrow['w']-.5,oldrow['h']-.5],(1000,2));native4=dict(newrow,transform_matrix=oldpose.tolist());ray_error=float(np.max(np.abs(ray(oldrow,uv)-ray(native4,uv*scale))));assert ray_error<1e-12
        image=a.source/newrow['file_path'];mask=a.source/newrow['mask_path'];assert Image.open(image).size==(newrow['w'],newrow['h']) and Image.open(mask).size==(newrow['w'],newrow['h'])
        image_hash=sha(image);mask_hash=sha(mask);assert image_hash==sources[camera]['rgb_sha256']
        if train:
            receipt_path=a.source/'receipts'/(camera+'.json');receipt=json.loads(receipt_path.read_text());assert sha(receipt_path)==complete['receipts'][camera]
            assert image_hash==receipt['output_hashes'][str(image)] and mask_hash==receipt['output_hashes'][str(mask)] and receipt['FHD_RGB_exact'] and receipt['FHD_semantics_exact']
            raw=receipt['raw'];replay=json.loads((a.source/'replay'/(camera+'.json')).read_text());assert replay['rgb_byte_exact'] and replay['valid_exact']
        else:
            assert image_hash==sha(old/oldrow['file_path']) and mask_hash==sha(old/oldrow['mask_path']);raw=None
        rows.append(dict(camera=camera,split='train' if train else 'eval',native_HD_camera=oldrow,prepared_4K_floor_camera=newrow,proposed_native_camera=native4,world_pose_roundtrip_maxabs=pose_error,continuous_center_ray_maxabs=ray_error,rgb_sha256=image_hash,valid_mask_sha256=mask_hash,raw=raw))
    remap={}
    for row in proposed['frames']:
        camera=row['physical_camera'];row['transform_matrix']=native[camera]['transform_matrix'];before=row['file_path'];row['file_path']=str(a.source/before);row['mask_path']=str(a.source/row['mask_path']);remap[before]=row['file_path']
    for split in ['train_filenames','val_filenames','test_filenames']:proposed[split]=[remap[p] for p in proposed[split]]
    (a.output/'proposed_transforms_NATIVE_WORLD_NOT_RELEASED.json').write_text(json.dumps(proposed,indent=2)+'\n')
    train_pixels=63*3840*2024;hd_pixels=63*1920*1012
    result=dict(kind='NATIVE_MIXED_RESOLUTION_SOURCE_FEASIBILITY_ONLY',training_authorized=False,source=str(a.source),source_training_eligible=complete['training_eligible'],source_status=complete['status'],native_source=str(old),rows=rows,bindings={str(p):sha(p) for p in [a.source/'transforms.json',a.source/'complete.json',a.source/'plan.json',a.source/'source_metadata.json',a.source/'post_export_array_verification.json',old/'transforms.json',old/'color_contract.json']},max_pose_roundtrip=max(r['world_pose_roundtrip_maxabs'] for r in rows),max_continuous_ray_difference=max(r['continuous_center_ray_maxabs'] for r in rows),proposed_transforms_sha256=sha(a.output/'proposed_transforms_NATIVE_WORLD_NOT_RELEASED.json'),memory_bytes=dict(HD_float32_RGB=hd_pixels*12,train4K_float32_RGB=train_pixels*12,train4K_bool_valid=train_pixels,train4K_RGB_plus_valid_collation_double=2*train_pixels*13),no_GPU_or_model_loaded=True,no_source_modified=True,limitations=['Parity is at continuous corresponding ray centers; integer4K pixels deliberately provide new subpixel rays.','No frequency maps or semantic actor objective are required for native; existing source eligibility remains unchanged.','Frozen native constructor bindings require a separately reviewed adapter before use.'])
    (a.output/'source_parity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['max_pose_roundtrip','max_continuous_ray_difference','memory_bytes']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
