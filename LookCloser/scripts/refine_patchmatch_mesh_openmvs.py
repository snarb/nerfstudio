#!/usr/bin/env python3
"""Opt-in fixed-camera photometric deformation of an existing TSDF mesh.

No SfM, pose optimization, masks, held-out RGB, remeshing or RGB synthesis.
The input mesh is copied to private scratch. Output is a refined TSDF-initialized
mesh, not a fresh TSDF extraction or a serialized raw TSDF volume.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import struct
import time
import numpy as np
from colmap_patchmatch_tsdf_campaign_common import atomic_json,append_jsonl,canonical_sha256,sha256
from export_nerfstudio_colmap_model import export_model
from render_patchmatch_camera_path import normalize_frame

PINNED_BINARIES={'RefineMesh':'627ba05bb972f3f50472e316ffb45cc9328167256098baf843826e038553f1b8',
                 'InterfaceCOLMAP':'c1f9947419da64db41db78734514e556da56ba29c1b775b6e54f9e5f927c5c1a',
                 'TransformScene':'cbc212e98d0c2d5d137f0f1357851637d6f87551427f31a08b0f1832ad38165e'}


def read_openmvs_no_points_images(path):
    """Read v2.4.0 --binary 1 --no-points 1's headerless image records.

    ExportScene skips assigning numRegImages when noPoints is true (upstream
    InterfaceCOLMAP.cpp), so WriteBIN omits the usual uint64 record count.
    Parse the documented per-record layout to EOF, requiring zero observations.
    Never use this reader for ordinary COLMAP binary models.
    """
    from nerfstudio.data.utils.colmap_parsing_utils import Image
    result={}
    with path.open('rb') as stream:
        while header:=stream.read(64):
            if len(header)!=64:raise ValueError('Truncated OpenMVS image record')
            row=struct.unpack('<i7di',header);name=bytearray()
            while True:
                char=stream.read(1)
                if not char or len(name)>65535:raise ValueError('Invalid OpenMVS image name')
                if char==b'\0':break
                name.extend(char)
            count=stream.read(8)
            if len(count)!=8 or struct.unpack('<Q',count)[0]!=0:
                raise ValueError('Expected no-points OpenMVS image record')
            if row[0] in result:raise ValueError('Duplicate OpenMVS image ID')
            result[row[0]]=Image(row[0],np.array(row[1:5]),np.array(row[5:8]),row[8],
                                 name.decode(),np.empty((0,2)),np.empty(0,dtype=np.int64))
    return result


def camera_inventory(model,*,openmvs_no_points=False):
    from nerfstudio.data.utils.colmap_parsing_utils import (
        read_cameras_text,read_images_text,read_cameras_binary,read_images_binary,qvec2rotmat)
    # OpenMVS text export uses only six significant digits. Audit its binary
    # export instead, so rounding cannot hide a real calibration change.
    if (model/'cameras.bin').exists() or (model/'images.bin').exists():
        cameras=read_cameras_binary(model/'cameras.bin')
        images=(read_openmvs_no_points_images(model/'images.bin') if openmvs_no_points
                else read_images_binary(model/'images.bin'))
    else:
        cameras=read_cameras_text(model/'cameras.txt');images=read_images_text(model/'images.txt')
    result={}
    for im in images.values():
        name=Path(im.name).name;cam=cameras[im.camera_id]
        if name in result or cam.model!='PINHOLE':raise ValueError('Duplicate image or changed pinhole model')
        result[name]=np.r_[cam.width,cam.height,cam.params,qvec2rotmat(im.qvec).ravel(),im.tvec]
    return result


def compare_calibration(before,after):
    if before.keys()!=after.keys():raise ValueError('Refinement changed calibrated image inventory')
    if not before or any(before[k].shape!=after[k].shape or not np.isfinite(before[k]).all()
                         or not np.isfinite(after[k]).all() for k in before):
        raise ValueError('Empty, malformed or nonfinite fixed calibration')
    error=max(float(np.abs(before[k]-after[k]).max()) for k in before)
    if error>1e-5:raise ValueError(f'Refinement changed fixed calibration: max difference {error}')
    return {'images':len(before),'max_absolute_difference':error,'poses_optimized':False}


def main():
    import open3d as o3d
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','mesh','mesh-metadata','openmvs-bin','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--max-threads',type=int,default=12)
    p.add_argument('--iterations',type=int,default=45,
                   help='Coarse-scale iterations; the second scale uses half this count (minimum 8).')
    args=p.parse_args()
    if args.iterations<8 or args.max_threads<1:p.error('Expected at least 8 iterations and positive thread count')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    for name,expected in PINNED_BINARIES.items():
        if sha256(args.openmvs_bin/name)!=expected:raise ValueError(f'Unverified OpenMVS 2.4.0 binary: {name}')
    payload=json.loads((args.data/'transforms.json').read_text());meta=json.loads(args.mesh_metadata.read_text())
    train=set(payload['train_filenames']);frames=[normalize_frame(f,payload,meta) for f in payload['frames'] if f['file_path'] in train]
    forbidden={'F004_B005_1210O9','J004_D005_1210TA','L004_B005_12106A'}
    if len(frames)!=62 or len({f['physical_camera'] for f in frames})!=62 or forbidden&{f['physical_camera'] for f in frames}:
        raise ValueError('Expected 62 unique train cameras, no held-out RGB')
    if any(f.get('mask_path') for f in payload['frames']):raise ValueError('Semantic masks forbidden')
    mesh_hash=sha256(args.mesh)
    if mesh_hash!=meta['output_sha256']:raise ValueError('Input mesh differs from normalization metadata')
    data=args.output/'data';(data/'images').mkdir(parents=True);source_hashes={}
    for f in frames:
        source=(args.data/f['file_path']).resolve(strict=True);name='images/'+source.name
        (data/name).symlink_to(source);source_hashes[f['physical_camera']]={'path':str(source),'sha256':sha256(source)}
        f['file_path']=name
    atomic_json(data/'transforms.json',{'frames':frames,'train_filenames':[f['file_path'] for f in frames]})
    model=args.output/'colmap/sparse'
    export_model(data,model,split='train',camera_model='PINHOLE')
    original_calibration=camera_inventory(model)
    initial=args.output/'input_mesh.ply';shutil.copyfile(args.mesh,initial)
    settings=['--resolution-level','0','--min-resolution','640','--max-views','8','--decimate','1',
              '--close-holes','0','--ensure-edge-size','0','--max-face-area','0','--scales','2','--scale-step','.5',
              '--regularity-weight','.2','--rigidity-elasticity-ratio','.9','--gradient-step',f'{args.iterations}.05',
              '--planar-vertex-ratio','0','--reduce-memory','1','--max-threads',str(args.max_threads),'--verbosity','2',
              '--archive-type','1']  # Force saving the refined scene for the final camera audit.
    request={'schema_version':1,'method':'fixed_camera_openmvs_photometric_mesh_refinement','input_mesh_sha256':mesh_hash,
             'metadata_sha256':sha256(args.mesh_metadata),'source_transforms_sha256':sha256(args.data/'transforms.json'),
             'normalized_transforms_sha256':sha256(data/'transforms.json'),'train_sources':source_hashes,
             'uses_eval_rgb':False,'uses_semantic_masks':False,'pose_optimization':False,'binary_sha256':PINNED_BINARIES,
             'openmvs_release':'v2.4.0','openmvs_commit':'58117204c86bbb11a0b25b26a8987676cf11274d',
             'settings':settings,'controller_sha256':sha256(Path(__file__)),'exporter_sha256':sha256(Path(__file__).with_name('export_nerfstudio_colmap_model.py'))}
    request['sha256']=canonical_sha256(request);atomic_json(args.output/'refinement_request.json',request)
    def run(command,stage):
        log=args.output/(stage+'.log')
        with log.open('w') as f:
            process=subprocess.Popen(list(map(str,command)),cwd=args.output,stdout=f,stderr=subprocess.STDOUT)
            last=0
            while process.poll() is None:
                if time.monotonic()-last>=60:
                    check={'timestamp':datetime.now(timezone.utc).isoformat(),'stage':stage,'controller_pid':os.getpid(),
                           'worker_pid':process.pid,'worker_alive':process.poll() is None,'output':str(args.output),
                           'disk_free_bytes':shutil.disk_usage(args.output).free}
                    with log.open('rb') as tail:
                        tail.seek(max(0,log.stat().st_size-1500));check['compact']=tail.read().decode(errors='replace')
                    check['gpu']=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
                    append_jsonl(args.output/'checks.jsonl',check)
                    print(f'stage={stage} pid={process.pid} active',flush=True);last=time.monotonic()
                time.sleep(2)
            if process.returncode:raise RuntimeError(f'{stage} failed; preserve scratch: {log}')
    interface=args.openmvs_bin/'InterfaceCOLMAP';scene=args.output/'scene.mvs'
    run([interface,'-i',model.parent,'-o',scene,'--image-folder',str(data)+'/', '--binary','0'], 'import')
    before=args.output/'roundtrip_before';before.mkdir()
    run([interface,'-i',scene,'-o',before,'--image-folder',str(data)+'/', '--binary','1','--no-points','1'],'roundtrip_before')
    import_check=compare_calibration(original_calibration,camera_inventory(before/'sparse',openmvs_no_points=True))
    refined=args.output/'refined.mvs'
    run([args.openmvs_bin/'RefineMesh','-i',scene,'-m',initial,'-o',refined,*settings],'refine')
    # InterfaceCOLMAP reads interchange archives only, not native Scene archives.
    # TransformScene requires an explicit operation; an identity transform keeps
    # every camera fixed. The original refined PLY, not its converted copy, is used.
    identity=args.output/'identity.txt';np.savetxt(identity,np.eye(4),fmt='%g')
    interchange=args.output/'refined_interface.mvs'
    run([args.openmvs_bin/'TransformScene','-i',refined,'-o',interchange,'--archive-type','-1',
         '--transform-file',identity,'--sample-mesh','0','--normalize-coordinates','0',
         '--max-threads',str(args.max_threads)],'convert_interface')
    after=args.output/'roundtrip_after';after.mkdir()
    run([interface,'-i',interchange,'-o',after,'--image-folder',str(data)+'/', '--binary','1','--no-points','1'],'roundtrip_after')
    pose_check=compare_calibration(original_calibration,camera_inventory(after/'sparse',openmvs_no_points=True))
    output_mesh=refined.with_suffix('.ply');before_mesh=o3d.io.read_triangle_mesh(str(initial));after_mesh=o3d.io.read_triangle_mesh(str(output_mesh))
    xyz=np.asarray(after_mesh.vertices);tri=np.asarray(after_mesh.triangles)
    if not len(tri) or not np.isfinite(xyz).all():raise ValueError('Empty/nonfinite refined mesh')
    if xyz.shape!=np.asarray(before_mesh.vertices).shape or not np.array_equal(tri,np.asarray(before_mesh.triangles)):
        raise ValueError('Topology changed despite disabled remeshing; inspect before rendering')
    displacement=np.linalg.norm(xyz-np.asarray(before_mesh.vertices),axis=1)
    labels,counts,_=after_mesh.cluster_connected_triangles()
    for row in source_hashes.values():
        if sha256(Path(row['path']))!=row['sha256']:raise ValueError('Source RGB changed during refinement')
    if sha256(args.mesh)!=mesh_hash or sha256(args.data/'transforms.json')!=request['source_transforms_sha256']:
        raise ValueError('Original mesh/source transforms changed')
    output_meta=deepcopy(meta);output_meta.update(method='tsdf_initialized_photometric_refined_mesh',output=str(output_mesh),
             output_sha256=sha256(output_mesh),refinement_request_sha256=request['sha256'],parent_mesh_sha256=mesh_hash,
             vertices=len(xyz),triangles=len(tri),connected_components=len(counts),component_triangles=sorted(map(int,counts),reverse=True))
    atomic_json(output_mesh.with_suffix('.json'),output_meta)
    result={'state':'refined_mesh_ready_for_visual_gate','request_sha256':request['sha256'],'mesh':str(output_mesh),
            'mesh_sha256':sha256(output_mesh),'mesh_metadata_sha256':sha256(output_mesh.with_suffix('.json')),
            'vertices':len(xyz),'triangles':len(tri),'components':len(counts),'topology_unchanged':True,
            'displacement_normalized':dict(zip(['min','median','p90','p99','max'],map(float,np.quantile(displacement,[0,.5,.9,.99,1])))),
            'import_calibration_check':import_check,'final_calibration_check':pose_check,'visual_status':'pending'}
    atomic_json(args.output/'refinement_result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
