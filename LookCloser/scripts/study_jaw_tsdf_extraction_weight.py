"""Matched TSDF extraction-weight controls using cached native geometric depths.

No new PatchMatch or camera/texture changes. Extracted control meshes are not
production repairs and not serialized raw TSDF volumes.
"""
from pathlib import Path
import os
from joint_temporal_texture import read, sha, atomic_json
from run_temporal_full_block_control import supervised

ROOT=Path('/mnt/data/dec5_jaw_tsdf_extraction_weight')
SOURCE=Path('/mnt/data/dec5_jaw_measured_depth/controls')
FRAMES=['001193','001195']
ARMS={'weight05':.5,'weight1':1.,'weight2':2.}


def main():
    ROOT.mkdir(exist_ok=False)
    scripts=['fuse_depth_tsdf_mesh.py','run_temporal_full_block_control.py',Path(__file__).name]
    request=dict(frames=FRAMES,arms=ARMS,only_changed_fusion_parameter='tensor_weight_threshold',
        scripts={n:sha(Path(__file__).with_name(n)) for n in scripts},
        cached_depths_only=True,new_patchmatch=False,production_changed=False,original_activation_mode=True,
        source_root=str(SOURCE),inferred_acceptance=False)
    atomic_json(ROOT/'request.json',request)
    os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',OPENCV_IO_ENABLE_OPENEXR='1')
    records=[]
    for frame in FRAMES:
        folder=ROOT/frame; folder.mkdir(); (folder/'stages').mkdir()
        source=SOURCE/frame; receipt=read(source/'stages/fuse-original.json')
        for p,h in receipt['retained_hashes'].items(): assert sha(p)==h,p
        base=receipt['command']; data=Path(base[base.index('--data')+1]); transforms=read(data/'transforms.json')
        train=set(transforms['train_filenames']); selected=[r for r in transforms['frames'] if r['file_path'] in train]
        assert len(train)==len(selected)==62 and len({r['physical_camera'] for r in selected})==62
        assert not ({r['physical_camera'] for r in selected}&{'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'})
        paths=[data/r['depth_file_path'] for r in selected]
        fq=dict(frame=frame,parent_request_sha256=sha(ROOT/'request.json'),
            reference_receipt_sha256=sha(source/'stages/fuse-original.json'),reference_command=base,
            transforms_sha256=sha(data/'transforms.json'),depth_hashes={str(p):sha(p) for p in paths},
            train_cameras=[r['physical_camera'] for r in selected],source_mesh_hashes=receipt['retained_hashes'])
        atomic_json(folder/'request.json',fq)
        for arm,weight in ARMS.items():
            dest=folder/arm; dest.mkdir(); command=list(base)
            command[command.index('--output')+1]=str(dest/'mesh.ply')
            command[command.index('--tensor-weight-threshold')+1]=str(weight)
            supervised(command,folder,'fuse-'+arm)
            meta=read(dest/'mesh.json'); records.append(dict(frame=frame,arm=arm,weight=weight,
                mesh_sha256=sha(dest/'mesh.ply'),metadata_sha256=sha(dest/'mesh.json')))
        for p,h in fq['depth_hashes'].items(): assert sha(p)==h,p
    atomic_json(ROOT/'result.json',dict(request_sha256=sha(ROOT/'request.json'),records=records,
        geometry_review_pending=True,production_accepted=False))
    print('Six cached-depth extraction controls complete',flush=True)


if __name__=='__main__': main()
