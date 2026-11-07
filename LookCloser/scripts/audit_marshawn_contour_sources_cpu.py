"""CPU provenance audit of actual removed/retained native contour layers.

Reuses TRAIN63 projection/support contracts and exact whole-cell voting code.
No model/GPU, new masks, physical-visibility labels, or producer changes.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from audit_marshawn_source_margin_cpu import admitted
from marshawn_semantic_cell_admission import WholeCellAdmission
from mask_visual_hull_aabb import MaskCamera,classify_cells,BITS


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def select_layers(w, keep):
    w=np.asarray(w,float).reshape(-1);keep=np.asarray(keep,bool).reshape(-1)
    if w.shape!=keep.shape or not np.isfinite(w).all() or (w<0).any():raise ValueError('Finite nonnegative matching weights required')
    result={}
    for label,selection in [('removed',~keep),('retained',keep)]:
        selected=np.flatnonzero(selection)
        if not len(selected)or w[selected].sum()<=0:raise ValueError('Both layers need observed contribution')
        result[label+'_peak']=int(selected[np.argmax(w[selected])])
        result[label+'_q50']=int(selected[np.searchsorted(np.cumsum(w[selected]),.5*w[selected].sum())])
    return result


def project(xyz,f):
    pose=np.asarray(f['transform_matrix']);v=(np.asarray(xyz)-pose[:3,3])@pose[:3,:3];z=-v[...,2]
    return np.stack([f['fl_x']*v[...,0]/z+f['cx'],f['cy']-f['fl_y']*v[...,1]/z],axis=-1),z


def main(c,out):
    out.mkdir(parents=True,exist_ok=False)
    trace=c/'independent_occupancy/quarter_distortion_source_margin_v1/quarter_distortion_raw126316/native_A/contour9_trace/rays';audit=json.loads((trace/'audit.json').read_text());assert audit['checkpoint_sha256']=='340e794ed8ea1cc656f246998941ddc9dee9d3a4244ee64952f2652ce35cd641'
    source=c/'native_semantic_grid_20000/occlusion_v2/W64';g=np.load(source/'semantic_grid.npz');mask=g['semantic_gate_pad1'];bounds=g['aabb'];pitch=(bounds[1]-bounds[0])/128
    manifest=json.loads((source/'support_manifest.json').read_text());support_rows={r['camera']:r for r in manifest['records']};meta=json.loads((c/'data/transforms.json').read_text());frames=[f for f in meta['frames']if f['file_path']in meta['train_filenames']];assert len(frames)==63
    points=[];inputs={str(trace/'audit.json'):sha(trace/'audit.json'),str(source/'semantic_grid.npz'):sha(source/'semantic_grid.npz'),str(source/'support_manifest.json'):sha(source/'support_manifest.json'),str(c/'data/transforms.json'):sha(c/'data/transforms.json')}
    for ri in [1,4]:
        row=audit['rows'][ri];q=row['pixel'];path=trace/row['profiles']['native']['trace'];z=np.load(path);inputs[str(path)]=sha(path);w=z['actual_weights'].reshape(-1);keep=admitted(z['xyz'],mask,bounds,margin=.002)
        for kind,i in select_layers(w,keep).items():
            xyz=z['xyz'][i];cell=np.floor((xyz-bounds[0])/pitch).astype(int);assert ((cell>=0)&(cell<128)).all();origin=np.asarray(row['origin']);direction=np.asarray(row['direction'])
            neighbors=np.array([cell+[dx,dy,dz]for dx in [-1,0,1]for dy in [-1,0,1]for dz in [-1,0,1]]);neighbors=neighbors[((neighbors>=0)&(neighbors<128)).all(1)]
            points.append(dict(label=q['camera_name']+'_'+kind,kind=kind,sample_index=i,pixel=q,xyz=xyz.tolist(),t=float((xyz-origin)@direction),sample_weight=float(w[i]),original_ray_alpha=float(w.sum()),removed_mass=float(w[~keep].sum()),retained_mass=float(w[keep].sum()),source_cell=cell.tolist(),source_cell_center=(bounds[0]+(cell+.5)*pitch).tolist(),gate0=bool(g['semantic_gate0'][tuple(cell)]),pad1=bool(mask[tuple(cell)]),margin002_admitted=bool(keep[i]),neighbor_cells=neighbors.tolist(),train_projections=[],eval_origin=origin.tolist()))
    unique=np.unique(np.array([i for p in points for i in p['neighbor_cells']]),axis=0);lower=bounds[0]+unique*pitch;upper=lower+pitch;cell_ids={tuple(i):n for n,i in enumerate(unique)};votes=[[]for _ in unique];bindings={}
    for f in frames:
        cam=f['physical_camera'];sr=support_rows[cam];sp=source/'archives'/f'{cam}.npz';mp=c/'data/semantics'/f'{cam}.npz';raw=Path(sr['source_archive']);assert sha(sp)==sr['archive_sha256'];assert sha(raw)==sr['source_archive_sha256']
        with np.load(sp)as z:s={k:z[k]for k in ['support','unknown','valid']}
        with np.load(mp)as z:m={k:z[k]for k in ['W64','core','band','valid','black','grey']}
        with np.load(raw)as z:original=z['W64']
        prepared=MaskCamera(cam,np.asarray(f['transform_matrix']),f['fl_x'],f['fl_y'],f['cx'],f['cy'],s['support'],s['unknown'],s['valid']).prepare(0)
        _,_,clear,_=classify_cells([prepared],lower,upper)
        for idx in np.flatnonzero(clear):votes[idx].append(cam)
        image=c/'data'/f['file_path'];bindings[cam]=dict(main_sha256=sha(mp),corrected_sha256=sha(sp),original_sha256=sha(raw),image=str(image),image_sha256=sha(image))
        for point in points:
            x=np.asarray(point['xyz']);uv,depth=project(x,f);px,py=np.floor(uv).astype(int);inside=depth>0 and 0<=px<f['w']and 0<=py<f['h'];cell=np.asarray(point['source_cell']);idx=cell_ids[tuple(cell)];corners=bounds[0]+(cell+BITS)*pitch;cuv,cz=project(corners,f);center_uv,center_z=project(point['source_cell_center'],f)
            r=dict(camera=cam,uv=uv.tolist(),camera_z=float(depth),in_frame=bool(inside),camera_distance=float(np.linalg.norm(x-np.asarray(f['transform_matrix'])[:3,3])),distance_to_eval_camera=float(np.linalg.norm(np.asarray(point['eval_origin'])-np.asarray(f['transform_matrix'])[:3,3])),whole_source_cell_clear_vote=bool(clear[idx]),source_cell_corner_uv=cuv.tolist(),source_cell_corner_camera_z=cz.tolist(),source_cell_center_uv=center_uv.tolist(),source_cell_center_camera_z=float(center_z))
            if inside:
                r.update(pixel=[int(px),int(py)],current_W64=float(m['W64'][py,px]),original_W64=float(original[py,px]),core=bool(m['core'][py,px]),band=bool(m['band'][py,px]),main_valid=bool(m['valid'][py,px]),black=bool(m['black'][py,px]),grey=bool(m['grey'][py,px]),support=bool(s['support'][py,px]),UNKNOWN=bool(s['unknown'][py,px]),source_valid=bool(s['valid'][py,px]));r['point_clear']=r['source_valid']and not r['support']and not r['UNKNOWN']
            point['train_projections'].append(r)
    for i,cell in enumerate(unique):
        assert (len(votes[i])<2)==bool(g['semantic_gate0'][tuple(cell)]),'Exact original gate0 votes disagree'
    for point in points:
        cell=np.asarray(point['source_cell']);idx=cell_ids[tuple(cell)];point['actual_gate0_veto_cameras']=votes[idx];point['pad1_neighbor_vetoes']=[dict(cell=cell2,cameras=votes[cell_ids[tuple(cell2)]])for cell2 in point['neighbor_cells']];assert any(len(x['cameras'])<2 for x in point['pad1_neighbor_vetoes'])==point['pad1']
        rows=[r for r in point['train_projections']if r['in_frame']];point['counts']={k:sum(bool(r[k])for r in rows)for k in ['core','band','black','grey','support','UNKNOWN','point_clear','whole_source_cell_clear_vote']};point['counts']['positive_current_W64']=sum(r['current_W64']>0 for r in rows)
        nearest=sorted(rows,key=lambda r:r['distance_to_eval_camera']);point['nearest_TRAIN_cameras_geometry_only']=[r['camera']for r in nearest[:5]]
        chosen=[]
        for condition in [lambda r:r['whole_source_cell_clear_vote'],lambda r:r['point_clear'],lambda r:r['core'],lambda r:True]:
            chosen.extend([r for r in nearest if condition(r)and r not in chosen][:2])
        chosen=chosen[:8];point['page_cameras']=[r['camera']for r in chosen];canvas=Image.new('RGB',(1200,500),'#151515');draw=ImageDraw.Draw(canvas);draw.text((4,3),point['label']+' xyz='+str(np.round(point['xyz'],5))+' t='+str(round(point['t'],5))+' | projection is NOT visibility',fill='yellow')
        for i,r in enumerate(chosen):
            px,py=r['pixel'];im=Image.open(bindings[r['camera']]['image']).convert('RGB');crop=im.crop((px-140,py-84,px+140,py+84));d=ImageDraw.Draw(crop);d.ellipse((132,76,148,92),outline='red',width=2)
            x=(i%4)*300;y=28+(i//4)*232;canvas.paste(crop,(x+10,y+52));draw.text((x+4,y),f"{r['camera']} Wmain {r['current_W64']:.3f} Worig {r['original_W64']:.3f}",fill='white');draw.text((x+4,y+14),f"core{int(r['core'])} grey{int(r['grey'])} SUP{int(r['support'])} UNK{int(r['UNKNOWN'])}",fill='white');draw.text((x+4,y+28),f"clearCell{int(r['whole_source_cell_clear_vote'])} clearPix{int(r['point_clear'])} UV{px},{py}",fill='white')
        canvas.save(out/(point['label']+'.jpg'),quality=95)
    result=dict(kind='CPU_ACTUAL_CONTOUR_LAYERS_TRAIN_SOURCE_CAUSAL_AUDIT',checkpoint_sha256=audit['checkpoint_sha256'],geometric_margin=.002,numerical_epsilon=2e-7,points=points,source_bindings=bindings,inputs_sha256=inputs,helper_sha256=sha(__file__),whole_cell_helper_sha256=sha(Path(__file__).with_name('mask_visual_hull_aabb.py')),reconstructed_source_gate0_cells=len(unique),all_reconstructed_gate0_matches=True,no_GPU=True,new_labels=False,limitations=['Discrete peak means maximum actual native sample weight, not fitted continuous mode.','Projection, source mask and nearest camera do not certify visibility. Source photos require manual inspection.','Veto is original whole-cell semantic evidence, not calibrated physical free-space truth.','Only two EVAL diagnostics choose positions; no EVAL supervision or new annotations.'])
    (out/'coverage.json').write_text(json.dumps(result,indent=2));print(json.dumps([dict(label=p['label'],xyz=p['xyz'],t=p['t'],removed=p['removed_mass'],retained=p['retained_mass'],votes=p['actual_gate0_veto_cameras'],counts=p['counts'])for p in points],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.campaign,a.output)
