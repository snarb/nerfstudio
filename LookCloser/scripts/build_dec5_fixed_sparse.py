#!/usr/bin/env python3
"""Fresh train59 triangulation under fixed known poses; no pose bundle adjustment."""
from pathlib import Path
import argparse,json,subprocess,time
import numpy as np
from export_nerfstudio_colmap_model import export_model
from audit_dec5_calibration import calibrate_database
from study_dec5_sfm_calibration import bind_database_ids
from prepare_dec5_lookcloser_sparse import prepare
from dec5_hair import cv_pose,VALIDATION,TEST
from run_dec5_hair_campaign import atomic_json,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['data','workspace','output','colmap']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.workspace.mkdir(parents=True,exist_ok=False);d=json.load(open(a.data/'transforms.json'));frames=[f for f in d['frames'] if f['file_path'] in d['train_filenames']]
    if len(frames)!=59 or any(f['physical_camera'] in VALIDATION+TEST for f in frames):raise ValueError('Train59 required')
    atomic_json(a.workspace/'prepare.json',dict(frames=frames,dataset_sha256=sha(a.data/'transforms.json'),train_only=True,inherited_sparse_points=False))
    export_model(a.data,a.workspace/'fixed_model',split='train',camera_model='PINHOLE')
    (a.workspace/'train_images.txt').write_text('\n'.join(f['file_path'] for f in frames)+'\n')
    centers=np.stack([cv_pose(f)[:3,3] for f in frames]);pairs=set()
    for i in range(len(frames)):
        for j in np.argsort(np.linalg.norm(centers-centers[i],axis=-1))[1:13]:pairs.add(tuple(sorted((frames[i]['file_path'],frames[j]['file_path']))))
    (a.workspace/'pairs.txt').write_text('\n'.join(x+' '+y for x,y in sorted(pairs))+'\n');db=a.workspace/'database.db';stages=[]
    def run(name,args):
        started=time.monotonic()
        with (a.workspace/(name+'.log')).open('w') as log:subprocess.run([str(a.colmap)]+list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,check=True)
        stages.append(dict(stage=name,seconds=time.monotonic()-started));atomic_json(a.workspace/'stages.json',stages)
    run('features',['feature_extractor','--database_path',db,'--image_path',a.data,'--image_list_path',a.workspace/'train_images.txt','--ImageReader.camera_model','PINHOLE','--SiftExtraction.use_gpu','1','--SiftExtraction.num_threads','2','--SiftExtraction.max_num_features','8192'])
    calibrate_database(a.workspace)
    run('matching',['matches_importer','--database_path',db,'--match_list_path',a.workspace/'pairs.txt','--match_type','pairs','--SiftMatching.use_gpu','1','--SiftMatching.num_threads','2'])
    bound=bind_database_ids(a.workspace);tri=a.workspace/'triangulated';tri.mkdir()
    run('triangulation',['point_triangulator','--database_path',db,'--image_path',a.data,'--input_path',bound,'--output_path',tri,'--Mapper.num_threads','2','--Mapper.fix_existing_frames','1','--Mapper.ba_refine_focal_length','0','--Mapper.ba_refine_extra_params','0','--Mapper.filter_max_reproj_error','8','--Mapper.tri_complete_max_reproj_error','8','--Mapper.tri_merge_max_reproj_error','8'])
    prepare(a.data,tri,a.output)

if __name__=='__main__':main()
