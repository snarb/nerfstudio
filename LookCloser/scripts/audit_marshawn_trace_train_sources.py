"""CPU-only train-GT/SfM/free-interval support for actual traced world samples."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from scipy.spatial import cKDTree,Delaunay
from PIL import Image,ImageDraw

ROOT=Path('/fsx/tmp/marshawn_001660_hd')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def project(frame,xyz):
    pose=np.asarray(frame['transform_matrix']);v=(np.asarray(xyz)-pose[:3,3])@pose[:3,:3];z=-v[2]
    return np.array([frame['fl_x']*v[0]/z+frame['cx'],frame['cy']-frame['fl_y']*v[1]/z]),float(z)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--trace-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=False)
    source=ROOT/'color_variants/gain4_scalar_trainfit_poseeval/sdr_png';meta=json.loads((source/'transforms.json').read_text());train=set(meta['train_filenames']);frames=[f for f in meta['frames']if f['file_path']in train];by={f['physical_camera']:f for f in frames};assert len(by)==63
    xyz=np.load(ROOT/'geometry/train_witnesses.npz')['xyz'];tree=cKDTree(xyz);obs=np.load(ROOT/'frequency_seed/observations.npz');free=np.load(ROOT/'free_space_candidates/scene_loose.npz');names=obs['camera_names'];assert len(obs['point_indices'])==44938
    fit=np.load(ROOT/'splatfacto_gain4/floor_geometry_diagnosis/manual_floor_fit.npz');n=fit['normal'];d=float(fit['offset']);support_ids=fit['point_indices'][fit['selected_inliers']];hull=Delaunay(xyz[support_ids,:2])
    inputs={};points=[]
    for variant in ['cold8k','uniform13k']:
        path=a.trace_root/variant/'audit.json';j=json.loads(path.read_text());inputs[variant]=dict(path=str(path),sha256=sha(path),checkpoint_sha256=j['checkpoint_sha256'])
        pixels=[[242,747],[300,740]]if variant=='cold8k'else[[242,747]]
        for row in j['rows']:
            if row['pixel']not in pixels:continue
            profiles=['native_core0','uniform32768']if variant=='cold8k'else['uniform32768']
            for profile in profiles:
                p=row['profiles'][profile]
                for kind,value in [('peak',p['peak']),('q50',next(v for v in p['termination_quantiles']if v['q']==.5))]:
                    if variant=='uniform13k'and kind=='peak':continue
                    points.append(dict(label=variant+'_'+str(row['pixel'][0])+'_'+str(row['pixel'][1])+'_'+profile+'_'+kind,
                        pixel=row['pixel'],profile=profile,kind=kind,xyz=value['xyz'],t=value['t'],sigma=value['sigma'],opacity=p['opacity']))
            if variant=='cold8k'and row['pixel']==[242,747]:
                origin=np.array(row['origin']);direction=np.array(row['direction']);t=-(origin@n+d)/(direction@n)
                points.append(dict(label='APPROX_PLANE_reference_242_747',pixel=row['pixel'],profile='not_a_field_sample',kind='extrapolated_approx_plane',xyz=(origin+t*direction).tolist(),t=float(t),sigma=None,opacity=None))
    rows=[];images={}
    for query in points:
        point=np.array(query['xyz']);distance,nearest=tree.query(point,k=8);nearest=[int(x)for x in nearest]
        priors=[]
        for pi,dist in zip(nearest,distance):
            ids=np.flatnonzero(obs['point_indices']==pi);priors.append(dict(point_idx=pi,distance=float(dist),xyz=xyz[pi].tolist(),signed_approx_plane_height=float(xyz[pi]@n+d),observations=len(ids),cameras=names[np.unique(obs['camera_indices'][ids])].tolist()))
        delta=point-free['origins'];t=(delta*free['directions']).sum(-1);clipped=np.minimum(np.maximum(t,free['t_start']),free['t_end']);closest=free['origins']+clipped[:,None]*free['directions'];gap=np.linalg.norm(point-closest,axis=1);free_ids=np.argsort(gap)[:8];intervals=[]
        for i in free_ids:
            oi=int(free['observation_idx'][i]);pi=int(free['point_idx'][i]);ci=int(free['camera_idx'][i]);assert pi==int(obs['point_indices'][oi])
            intervals.append(dict(archive_row=int(i),observation_idx=oi,point_idx=pi,camera=str(free['camera_names'][ci]),segment_distance=float(gap[i]),unclipped_t=float(t[i]),free_start=float(free['t_start'][i]),free_end=float(free['t_end'][i]),witness_depth=float(free['witness_depth'][i]),inside_free_interval=bool(free['t_start'][i]<t[i]<free['t_end'][i]),witness_xyz=xyz[pi].tolist(),witness_signed_approx_plane_height=float(xyz[pi]@n+d),calibrated_reprojected_track_uv=obs['uv'][oi].tolist()))
        projections=[]
        for name,frame in by.items():
            uv,z=project(frame,point);inside=z>0 and 0<=uv[0]<frame['w']and 0<=uv[1]<frame['h']
            entry=dict(camera=name,uv=uv.tolist(),camera_z=z,in_frame=bool(inside),camera_distance=float(np.linalg.norm(point-np.array(frame['transform_matrix'])[:3,3])))
            projections.append(entry)
        visible=[v for v in projections if v['in_frame']];visible.sort(key=lambda v:v['camera_distance']);preferred=list(dict.fromkeys(priors[0]['cameras']+[v['camera']for v in intervals]+[v['camera']for v in visible]));chosen=[next(v for v in visible if v['camera']==name)for name in preferred if any(v['camera']==name for v in visible)][:12]
        panels=[]
        for v in chosen:
            name=v['camera'];frame=by[name]
            if name not in images:images[name]=Image.open(source/frame['file_path']).convert('RGB')
            u,w=v['uv'];box=(int(round(u))-128,int(round(w))-96,int(round(u))+128,int(round(w))+96);crop=images[name].crop(box);draw=ImageDraw.Draw(crop);draw.ellipse((123,91,133,101),outline='red',width=2)
            track=np.flatnonzero((obs['point_indices']==nearest[0])&(names[obs['camera_indices']]==name))
            if len(track):
                observed=obs['uv'][track[0]];x,y=observed-np.array(box[:2]);draw.line((x-5,y,x+5,y),fill='cyan',width=2);draw.line((x,y-5,x,y+5),fill='cyan',width=2)
            panel=Image.new('RGB',(300,240),'#151515');panel.paste(crop,(22,45));pd=ImageDraw.Draw(panel);pd.text((3,3),name+' actualTRAIN; red=tracedXYZ',fill='white');pd.text((3,17),f'nearest p{nearest[0]} d3D={distance[0]:.4f}',fill='white');pd.text((3,31),'cyan=reprojected tracked point'if len(track)else'nearest point not observed by this cam',fill='cyan'if len(track)else'gray');panels.append(panel)
        canvas=Image.new('RGB',(1200,760),'black');draw=ImageDraw.Draw(canvas);draw.text((3,3),query['label']+' | frustum inclusion != visibility; approximate plane != depth truth',fill='yellow')
        for i,panel in enumerate(panels):canvas.paste(panel,((i%4)*300,(i//4)*240+30))
        page=a.output/(query['label']+'.jpg');canvas.save(page,quality=94)
        rows.append(dict(query,train_frustum_count=len(visible),signed_approx_plane_height=float(point@n+d),inside_plane_support_XY_hull=bool(hull.find_simplex(point[:2])>=0),nearest_witnesses=priors,nearest_trusted_free_segments=intervals,all63_train_projections=projections,source_GT_page=page.name,source_GT_page_sha256=sha(page)))
    report=dict(trace_sources=inputs,points_source_sha256=sha(ROOT/'geometry/train_witnesses.npz'),observations_sha256=sha(ROOT/'frequency_seed/observations.npz'),free_archive_sha256=sha(ROOT/'free_space_candidates/scene_loose.npz'),transforms_sha256=sha(source/'transforms.json'),plane_sha256=sha(ROOT/'splatfacto_gain4/floor_geometry_diagnosis/manual_floor_fit.npz'),plane_is_approximate=True,uses_train_images_only=True,no_model_loaded=True,optimizer_updates=0,rows=rows,limitations=['Source crops and frustum inclusion do not independently prove visibility or depth.','Track-camera IDs are actual SfM graph memberships, but cached UV are calibrated XYZ projections, not raw detected feature locations.','Nearest witness/free-ray proximity is a prior association clue, not proof of the cause of learned density.','Approximate15point cloth plane may be extrapolated and is never a carving or supervision rule.'])
    (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print([(r['label'],r['train_frustum_count'],r['signed_approx_plane_height'],r['nearest_witnesses'][0]['point_idx'],r['nearest_witnesses'][0]['distance'])for r in rows])


if __name__=='__main__':main()
