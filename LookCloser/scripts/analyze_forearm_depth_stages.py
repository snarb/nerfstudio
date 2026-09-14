"""Locate real-train forearm losses before/after geometric filtering and TSDF.

The manual region is traced on train RGB, not on a candidate's valid surface.
Counts diagnose processing stages; they are not ground-truth quality metrics.
"""
from pathlib import Path
import argparse
import numpy as np
import cv2
import open3d as o3d
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json,cameras
from render_patchmatch_camera_path import normalize_frame
from import_colmap_mvs_depth_dataset import read_colmap_dense_array
from study_confidence_depth_prior import raycast_integer,unproject,support,project_integer
from diffusion_mesh_repair import scene_for

POLYGONS={
    'forearm':[(211,1769),(308,1774),(281,1904),(186,1904),(193,1824)],
    'palm':[(140,1583),(275,1569),(310,1650),(314,1718),(289,1741),(197,1717),(143,1662)],
}


def stats(values):
    if not len(values):return dict(count=0,median=None,p10=None,p90=None)
    return dict(count=len(values),median=float(np.median(values)),
                p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)))


def analyze(control, output):
    frame='001033';probe=Path('/mnt/data/dec5_forearm_train_probe')/frame
    reference=read(probe/'references.json');entry=next(x for x in reference['evidence'] if x['camera']['physical_camera']=='G004_B005_1210FG')
    ref=entry['camera'];rgbpath=probe/'train_1.png'
    if sha(rgbpath)!=entry['image_sha256']:raise ValueError('Manual ROI source changed')
    metadata=read(reference['metadata']);rows,_,_=cameras(frame)
    spec=read(control/'staged63/transforms.json');lookup={r['physical_camera']:r for r in spec['frames']}
    depthdir=control/'pipeline/dense/stereo/depth_maps';depths=[];hashes={}
    for row in rows:
        raw=lookup[row['physical_camera']];mapped=normalize_frame(raw,spec,metadata)
        for key in ['transform_matrix','fl_x','fl_y','cx','cy']:
            if not np.allclose(mapped[key],row[key],rtol=0,atol=1e-6):raise ValueError('Depth gauge mismatch')
        path=depthdir/(raw['file_path']+'.geometric.bin');d=read_colmap_dense_array(path)
        if d.shape!=(1080,1920,1) or not np.isfinite(d).all():raise ValueError('Bad depth')
        depths.append(d[...,0]*metadata['dataparser_scale']);hashes[str(path)]=sha(path)
    index=next(i for i,r in enumerate(rows) if r['physical_camera']==ref['physical_camera']);geo=depths[index]
    photopath=depthdir/(lookup[ref['physical_camera']]['file_path']+'.photometric.bin')
    photo=read_colmap_dense_array(photopath)[...,0]*metadata['dataparser_scale'];hashes[str(photopath)]=sha(photopath)
    mesh=o3d.io.read_triangle_mesh(reference['mesh']);md=raycast_integer(scene_for(np.asarray(mesh.vertices),np.asarray(mesh.triangles)),ref)
    output.mkdir(parents=True,exist_ok=True);binding=dict(frame=frame,reference_rgb_sha256=sha(rgbpath),
        original_mesh_sha256=sha(reference['mesh']),control_request_sha256=sha(control/'request.json'),
        source_depth_hashes=hashes,regions=POLYGONS,script_sha256=sha(__file__),manual_train_rgb_only=True)
    if (output/'request.json').exists() and read(output/'request.json')!=binding:raise ValueError('Immutable analysis mismatch')
    atomic_json(output/'request.json',binding)
    rgb=np.asarray(Image.open(rgbpath).convert('RGB'));parts=[rgb.copy() for _ in range(4)];result={};maps={}
    for name,polygon in POLYGONS.items():
        mask=np.zeros((1920,1080),np.uint8);cv2.fillPoly(mask,[np.array(polygon,np.int32)],1);mask=np.rot90(mask,-1).astype(bool)
        pm=mask&(geo>0);y,x=np.nonzero(pm);points=unproject(ref,x,y,geo[y,x]);counts,free=support(points,ref,rows,depths)
        votes=np.zeros(md.shape,np.uint8);votes[y,x]=counts
        visible=mask&(md>0);missing=mask&(md==0)
        result[name]=dict(pixels=int(mask.sum()),original_mesh_missing=int(missing.sum()),
            photometric_valid=int((mask&(photo>0)).sum()),geometric_valid=int(pm.sum()),
            valid_geometric_but_missing_mesh=int((missing&pm).sum()),
            missing_mesh_with_two_other_depth_votes=int((missing&pm&(votes>=2)).sum()),
            other_geometric_votes=stats(counts),geometric_depth=stats(geo[pm]),
            original_mesh_depth=stats(md[visible]),photometric_depth_on_mesh_misses=stats(photo[missing&(photo>0)]))
        for i,condition in [(1,missing),(2,mask&(geo==0)),(3,mask&(votes>=2)&(md==0))]:
            view=np.rot90(condition);parts[i][view]=[255,30,0] if i<3 else [0,220,0]
        for part in parts:cv2.polylines(part,[np.array(polygon,np.int32)],True,(255,255,0),2)
        maps[name+'_region']=mask;maps[name+'_other_votes']=votes
    panel=Image.new('RGB',(1600,540));draw=ImageDraw.Draw(panel)
    for i,(label,im) in enumerate(zip(['Real train RGB','RED original mesh misses','RED geometric depth misses','GREEN >=2 other depths but no mesh'],parts)):
        Image.fromarray(im).save(output/f'overlay_{i}.png')
        crop=Image.fromarray(im).crop((60,1410,460,1920));panel.paste(crop,(i*400,30));draw.text((i*400+3,5),label,fill='white')
    panel.save(output/'comparison.png');np.savez_compressed(output/'evidence.npz',**maps)
    atomic_json(output/'result.json',dict(regions=result,mesh_pixel_convention='integer camera rays matched to COLMAP',
        depth_votes_require_reprojection_and_parallax=True,image_quality_metrics_computed=False,visual_status='pending'))
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',type=Path,default=Path('/mnt/data/dec5_forearm_depth_control_001033'))
    p.add_argument('--output',type=Path,default=Path('/mnt/data/dec5_forearm_depth_control_001033/depth_stage_analysis'))
    a=p.parse_args();analyze(a.control,a.output)
