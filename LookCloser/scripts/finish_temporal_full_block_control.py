"""Verify and recover a completed depth import after NFS copy2 metadata failure.

Strictly limited to the known terminal copystat failure after transforms/depth
publication. Numeric arrays are checked against their original COLMAP sources;
no partial data or a mere output filename can qualify as a recovered import.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import gzip
import numpy as np
from joint_temporal_texture import read,sha,atomic_json
from import_colmap_mvs_depth_dataset import read_colmap_dense_array
from run_temporal_full_block_control import supervised


def finish(output):
    import fcntl
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        request=read(output/'request.json');scripts=Path(__file__).parent
        for name in ['fuse_depth_tsdf_mesh.py','import_colmap_mvs_depth_dataset.py']:
            if sha(scripts/name)!=request['scripts'][name]:raise ValueError('Numeric implementation changed')
        log=(output/'logs/import-depth.log').read_text()
        if 'shutil.copy2' not in log or 'copystat' not in log or 'PermissionError: [Errno 1] Operation not permitted' not in log:
            raise ValueError('This recovery is only for the diagnosed terminal metadata-copy error')
        data=output/'pipeline/depth_dataset';payload=read(data/'transforms.json');summary=payload['colmap_mvs_depth']
        if summary['train_depth_count']!=62 or summary['heldout_invalid_depth_count']!=1 or summary['depth_shapes']!=[[1080,1920]]:
            raise ValueError('Imported depth inventory is incomplete')
        if sha(data/'transforms.source.json')!=sha(output/'staged63/transforms.json'):
            raise ValueError('Copy failure affected actual source-camera bytes')
        coverage=[];hashes={}
        for row in summary['depth_maps']:
            source,target=Path(row['source']),Path(row['output'])
            if sha(source)!=row['source_sha256'] or sha(target)!=row['output_sha256']:raise ValueError('Depth checksum mismatch')
            dense=read_colmap_dense_array(source)
            if dense.shape!=(1080,1920,1):raise ValueError('Expected scalar full-resolution COLMAP map')
            dense=dense[...,0];valid=np.isfinite(dense)&(dense>0)
            with gzip.open(target,'rb') as stream:converted=np.load(stream,allow_pickle=False)
            if not np.array_equal(converted,np.where(valid,dense,0).astype(np.float32)) or not valid.any():raise ValueError('Imported numeric depth changed')
            hashes[str(target)]=sha(target);coverage.append(float(valid.mean()))
        train=set(payload['train_filenames'])
        for row in payload['frames']:
            image=data/row['file_path']
            if not image.is_file():raise ValueError('Broken imported image link')
            if row['file_path'] not in train:
                with gzip.open(data/row['depth_file_path'],'rb') as stream:invalid=np.load(stream,allow_pickle=False)
                if np.any(invalid):raise ValueError('Held-out depth must remain invalid')
        for p in data.rglob('*'):
            if p.is_file() and p.suffix in ['.json','.gz']:hashes[str(p)]=sha(p)
        recovery={'method':'validate complete numeric output; ignore unsupported file timestamp preservation',
                  'original_import_exit':'failed only at terminal copy2/copystat',
                  'request_sha256':sha(output/'request.json'),'recovery_script_sha256':sha(__file__),
                  'scalar_colmap_shape':[1080,1920,1],'squeezed_depth_shape':[1080,1920],
                  'validated_real_maps':62,'lossless_import_comparison':True,'hashes':hashes}
        atomic_json(output/'import_recovery.json',recovery)
        commands=read(output/'commands.json');command=next(c for n,c in commands if n=='import-depth')
        atomic_json(output/'stages/import-depth-recovered.json',{'command':command,'request_sha256':sha(output/'request.json'),
                    'retained_hashes':hashes,'recovery_sha256':sha(output/'import_recovery.json')})
        atomic_json(output/'depth_qc.json',{'maps':62,'shape':[1080,1920],'coverage_mean':float(np.mean(coverage)),
                    'coverage_min':float(np.min(coverage)),'import_recovery_sha256':sha(output/'import_recovery.json')})
        fuse=next(c for n,c in commands if n=='fuse-tsdf')
        for name,extra in [('fuse-original',[]),('fuse-full-block',['--tensor-full-block-integration'])]:
            command=list(fuse);folder=output/name;folder.mkdir(exist_ok=True)
            command[command.index('--output')+1]=str(folder/'mesh.ply');supervised(command+extra,output,name)
        paths=['fuse-original/mesh.ply','fuse-original/mesh.json','fuse-full-block/mesh.ply','fuse-full-block/mesh.json','depth_qc.json','import_recovery.json']
        atomic_json(output/'complete.json',{'request_sha256':sha(output/'request.json'),'visual_status':'pending',
                    'completed_by_recovery_script_sha256':sha(__file__),'hashes':{p:sha(output/p) for p in paths}})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();finish(a.output)
