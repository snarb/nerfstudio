"""Project actual saved native artifact samples into TRAIN sources, read-only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from marshawn_semantic_cell_admission import WholeCellAdmission


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(c,trace,out):
    out.mkdir(parents=True,exist_ok=False)
    meta=json.loads((c/'data/transforms.json').read_text());train=set(meta['train_filenames']);frames=[f for f in meta['frames'] if f['file_path'] in train];assert len(frames)==63
    manifest=json.loads((c/'semantic_cold_quarterstep_candidate_20261107/source/support_manifest.json').read_text());support_rows={r['camera']:r for r in manifest['records']}
    audit=json.loads((trace/'audit.json').read_text());assert audit['checkpoint_sha256']=='542e97f59ec1fd41452b0a04fde6bd832e0b430773d1989eef2ec062ef3f1a34'
    targetpath=c/'clear_targets_seven_views_v1/targets_10481.json'
    if not targetpath.exists():targetpath=c/'quarter_seven_tau_candidate_v1/targets_10481.json'
    targets=json.loads(targetpath.read_text())['targets'];assert len(targets)==10481
    origin=np.array([t['origin'] for t in targets]);direction=np.array([t['direction'] for t in targets]);lo=np.array([t['near']for t in targets]);hi=np.array([t['far']for t in targets]);targetset={(t['camera'],t['x'],t['y'])for t in targets}
    points=[]
    for ri,row in enumerate(audit['rows']):
        z=np.load(trace/row['trace']);w=z['actual_weights'].reshape(-1);xyz=z['xyz'];assert len(w)>0
        cdf=np.cumsum(w);ids={'peak':int(w.argmax()),'q50':int(np.searchsorted(cdf,.5*cdf[-1]))}
        for kind,i in ids.items():points.append(dict(label=row['pixel']['label']+'_'+kind,ray=ri,kind=kind,xyz=xyz[i].tolist(),actual_sample_weight=float(w[i]),ray_alpha=row['alpha'],pixel=row['pixel']))
    sealed=c/'semantic_cold_quarterstep_candidate_20261107/source/semantic_grid.npz';g=np.load(sealed)
    gates={key:WholeCellAdmission(g[key],g['aabb']) for key in ['semantic_gate0','semantic_gate_pad1']}
    mapped=gates['semantic_gate_pad1'].map_target_grid((128,128,128),g['aabb'])
    sealed_manifest=json.loads((sealed.parent/'manifest.json').read_text());assert hashlib.sha256(mapped.tobytes()).hexdigest()==sealed_manifest['mapped_bool_sha256']
    for point in points:
        x=np.asarray(point['xyz']);t=((x-origin)*direction).sum(-1);clipped=np.clip(t,lo,hi);gap=np.linalg.norm(x-origin-clipped[:,None]*direction,axis=-1);near=int(gap.argmin())
        ijk=np.clip(np.floor((x-g['aabb'][0])/(g['aabb'][1]-g['aabb'][0])*128).astype(int),0,127)
        point['admission']=dict({key:bool(gate.point_admitted(x[None])[0])for key,gate in gates.items()},mapped_LC_cell=bool(mapped[tuple(ijk)]),LC_cell_index=ijk.tolist(),source_grid_sha256=sha(sealed),convention='closed-boundary epsilon2e-7 WholeCellAdmission; XYZ already accepted floor frame')
        point['nearest_10481_segment']=dict(index=near,distance=float(gap[near]),target=targets[near],inside_interval=bool(lo[near]<=t[near]<=hi[near]))
        point['train_projections']=[]
    sourcebindings={}
    # Open one source archive at a time; no model or labels are created.
    for frame in frames:
        name=frame['physical_camera'];pose=np.array(frame['transform_matrix']);s=c/'native_semantic_grid_20000/occlusion_v2/W64/archives'/f'{name}.npz';m=c/'data/semantics'/f'{name}.npz'
        assert sha(s)==support_rows[name]['archive_sha256']
        raw=Path(support_rows[name]['source_archive']);assert sha(raw)==support_rows[name]['source_archive_sha256']
        with np.load(s) as z:support={k:z[k] for k in ['support','unknown','valid']}
        with np.load(m) as z:main={k:z[k] for k in ['W64','core','band','valid']}
        with np.load(raw) as z:original={'W64':z['W64']}
        sourcebindings[name]=dict(corrected_support_sha256=sha(s),original_confidence_path=str(raw),original_confidence_sha256=sha(raw),main_semantics_sha256=sha(m),source_image=str(c/'data'/frame['file_path']),source_image_sha256=sha(c/'data'/frame['file_path']))
        for point in points:
            v=(np.asarray(point['xyz'])-pose[:3,3])@pose[:3,:3];z=-v[2];uv=[frame['fl_x']*v[0]/z+frame['cx'],frame['cy']-frame['fl_y']*v[1]/z];xx,yy=np.floor(uv).astype(int);valid=z>0 and 0<=xx<frame['w'] and 0<=yy<frame['h']
            row=dict(camera=name,uv=uv,in_frame=bool(valid),camera_z=float(z))
            if valid:
                row.update(pixel=[int(xx),int(yy)],W64=float(main['W64'][yy,xx]),original_confidence_W64=float(original['W64'][yy,xx]),main_core=bool(main['core'][yy,xx]),main_band=bool(main['band'][yy,xx]),main_valid=bool(main['valid'][yy,xx]),qualified_support=bool(support['support'][yy,xx]),UNKNOWN=bool(support['unknown'][yy,xx]),source_valid=bool(support['valid'][yy,xx]),in_current10481_pixel=(name,int(xx),int(yy)) in targetset)
                row['qualified_clear_pixel']=row['source_valid'] and not row['qualified_support'] and not row['UNKNOWN']
            point['train_projections'].append(row)
    for point in points:
        rows=[r for r in point['train_projections']if r['in_frame']]
        point['counts']={k:sum(bool(r[k])for r in rows)for k in ['main_core','main_band','qualified_support','UNKNOWN','qualified_clear_pixel','in_current10481_pixel']}
        point['counts'].update(in_frame=len(rows),main_positive=sum(r['main_valid']and r['W64']>0 for r in rows),positive_soft_band=sum(r['main_valid']and 0<r['W64']<1 for r in rows),main_zero=sum(r['main_valid']and r['W64']==0 for r in rows))
        chosen=[]
        for condition in [lambda r:r['qualified_clear_pixel']and r['W64']==0,lambda r:0<r['W64']<1,lambda r:r['UNKNOWN'],lambda r:r['main_core']]:
            selected=[r for r in rows if condition(r) and r not in chosen][:2];chosen.extend(selected)
        chosen.extend([r for r in rows if r not in chosen][:8-len(chosen)]);chosen=chosen[:8]
        canvas=Image.new('RGB',(960,420),'#111111');draw=ImageDraw.Draw(canvas);draw.text((3,2),point['label']+' XYZ '+str(np.round(point['xyz'],5))+' | red=projection; inclusion != visibility',fill='yellow')
        for i,r in enumerate(chosen):
            xx,yy=r['pixel'];im=Image.open(sourcebindings[r['camera']]['source_image']).convert('RGB');crop=im.crop((xx-110,yy-72,xx+110,yy+72));d=ImageDraw.Draw(crop);d.ellipse((104,66,116,78),outline='red',width=2)
            x=(i%4)*240;y=24+(i//4)*195;canvas.paste(crop,(x+10,y+42));d=ImageDraw.Draw(canvas);d.text((x+3,y),f"{r['camera']} W64={r['W64']:.3f} core={int(r['main_core'])}",fill='white');d.text((x+3,y+14),f"SUP={int(r['qualified_support'])} UNK={int(r['UNKNOWN'])} clr={int(r['qualified_clear_pixel'])}",fill='white');d.text((x+3,y+28),f"pixel {xx},{yy}; 10481={r['in_current10481_pixel']}",fill='white')
        page=out/(point['label']+'.jpg');canvas.save(page,quality=94);point['source_page']=page.name;point['source_page_sha256']=sha(page)
    result=dict(kind='ACTUAL_NATIVE_ARTIFACT_TRAIN_SOURCE_PROJECTION',checkpoint_sha256=audit['checkpoint_sha256'],trace_audit_sha256=sha(trace/'audit.json'),transforms_sha256=sha(c/'data/transforms.json'),targets_sha256=sha(targetpath),source_bindings=sourcebindings,points=points,limitations=['Frustum or mask inclusion does not prove physical visibility or depth.','Qualified clear source pixel establishes only a candidate visible prefix before its first physical surface, never an entire empty 3D ray.','EVAL pixels only choose diagnostic rays; no training labels generated.','Point source evidence alone does not allocate causation among positive RGB supervision, unobserved density, and other rays.'])
    (out/'coverage.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([dict(label=p['label'],z=p['xyz'][2],counts=p['counts'],nearest_segment=p['nearest_10481_segment']['distance'])for p in points],indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True);ap.add_argument('--trace',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();main(a.campaign,a.trace,a.output)
