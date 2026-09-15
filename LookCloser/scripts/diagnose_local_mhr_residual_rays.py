"""Read-only staged ray attribution for a sealed local MHR completion.

Portrait pixels are posthoc diagnostics, never a fit or admission domain.
Every stage uses exact frozen triangles and the production pinhole rays.
"""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw
from run_local_mhr_completion import read,save,sha,require
from review_local_mhr_transfer import enclosed_misses
from admit_mhr_local_patch_depth import Scene2


def landscape_pixel(x,y,width=1920):
    return width-1-y,x


def first_absent_stage(hits):
    names=['prior_full','anatomical_band','safe_band','local_before_centroid',
           'raw_candidates','semantic_candidates']
    return next((name for name in names if not hits[name]),'depth_or_native_admission')


def main():
    import open3d as o3d
    p=argparse.ArgumentParser();p.add_argument('--completion',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.completion;out=a.output;require(not out.exists(),'New output required')
    seal=read(root/'final_seal.json');require(seal['status']=='passed','Unsealed completion')
    inputs={str(root/'final_seal.json'):sha(root/'final_seal.json')}
    for path,h in seal['checked_bindings'].items():require(sha(path)==h,'Changed bound input');inputs[path]=h
    for path,h in seal['inventory'].items():require(sha(root/path)==h,'Changed sealed output');inputs[str(root/path)]=h
    config=read(root/'config.json');spec=config['spec'];frame=spec['frame']
    cq=read(root/'candidates/request.json');prior=Path(cq['final_prior_root'])
    fit=np.load(prior/'fit.npz');v,t,n=fit['vertices'],fit['triangles'],fit['neutral']
    dom=np.load(root/'candidates/silhouette100/domain_evidence.npz')
    prop=np.load(root/'candidates/silhouette100/proposal_evidence.npz')
    admission=np.load(root/'admission/silhouette100/admission.npz')
    review=read(root/'native_rgb/current_moving.json');camera=review['crops']['current_moving']['camera']
    base=root/'admission/rgb/current_moving/baseline/frames'/frame
    rgb=np.array(Image.open(base/'frame.png'));saved=np.load(base/'target_depth.npz')['depth']
    portrait=np.rot90(saved);y,x=np.nonzero(enclosed_misses(portrait,review['crops']['current_moving']['crop']))
    xy=np.c_[x,y];require(len(xy)>0,'No residual enclosed misses')
    original=o3d.io.read_triangle_mesh(cq['source_mesh'])
    ov,ot=np.asarray(original.vertices),np.asarray(original.triangles)
    raw=o3d.io.read_triangle_mesh(str(root/'candidates/silhouette100/local_raw.ply'))
    rv=np.asarray(raw.vertices);pp=prop['proposals'];parent=prop['parent_triangle_ids']
    band=((n[t,1]>=135)&(n[t,1]<=153)).all(1);safe=band&~dom['unsafe_parent']
    sem=admission['semantic_ids'];stages={}
    def add(name,vertices,triangles,ids,prior_ids):
        stages[name]=dict(vertices=vertices,triangles=triangles,ids=np.asarray(ids),prior_ids=np.asarray(prior_ids),scene=Scene2(vertices,triangles))
    add('original',ov,ot,np.arange(len(ot)),np.full(len(ot),-1))
    for name,keep in [('prior_full',np.ones(len(t),bool)),('anatomical_band',band),('safe_band',safe)]:
        ids=np.flatnonzero(keep);add(name,v,t[ids],ids,ids)
    ids=np.flatnonzero(dom['local_before_centroid_gate'])
    add('local_before_centroid',dom['subdivided_vertices'],dom['subdivided_triangles'][ids],ids,dom['parent_ids'][ids])
    add('raw_candidates',rv,pp,np.arange(len(pp)),parent)
    add('semantic_candidates',rv,pp[sem],sem,parent[sem])
    for branch in ['strict','interpolated']:
        ids=sem[admission[branch]];add(branch+'_initial',rv,pp[ids],ids,parent[ids])
        ids=np.load(root/'admission/silhouette100'/branch/'evidence.npz')['retained_proposal_ids']
        add(branch+'_final',rv,pp[ids],ids,parent[ids])
    pose=np.asarray(camera['transform_matrix']);ext=np.linalg.inv(pose@np.diag([1.,-1.,-1.,1.])).astype(np.float32)
    k=np.array([[camera['fl_x'],0,camera['cx']],[0,camera['fl_y'],camera['cy']],[0,0,1]],np.float32)
    rays=stages['original']['scene'].create_rays_pinhole(o3d.core.Tensor(k),o3d.core.Tensor(ext),camera['w'],camera['h']).numpy()
    xx,yy=landscape_pixel(x,y,camera['w']);query=rays[yy,xx]
    # Assert that these are exact current-camera original misses, not rotated/indexed substitutes.
    check=stages['original']['scene'].cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
    np.testing.assert_array_equal(np.where(np.isfinite(check),check,0),saved)
    out.mkdir();records=[dict(portrait_xy=q.tolist(),landscape_xy=[int(u),int(w)],ray_origin=r[:3].tolist(),ray_direction=r[3:].tolist(),stages={}) for q,u,w,r in zip(xy,xx,yy,query)]
    arrays=dict(portrait_xy=xy,landscape_xy=np.c_[xx,yy],rays=query)
    for name,s in stages.items():
        h=s['scene'].cast_rays(o3d.core.Tensor(query));d=h['t_hit'].numpy();tid=h['primitive_ids'].numpy();uv=h['primitive_uvs'].numpy()
        arrays[name+'_depth']=d;arrays[name+'_triangle_local']=tid;arrays[name+'_uv']=uv
        for i in range(len(xy)):
            hit=bool(np.isfinite(d[i]));q=dict(hit=hit,depth=float(d[i]) if hit else None,scene_triangle_id=int(tid[i]) if hit else None)
            if hit:
                j=int(tid[i]);pid=int(s['prior_ids'][j]);q.update(stage_triangle_id=int(s['ids'][j]),prior_parent_triangle_id=pid,
                    barycentric_uv=uv[i].tolist(),point=(query[i,:3]+d[i]*query[i,3:]).tolist())
                if pid>=0:q['neutral_parent_y_cm']=n[t[pid],1].tolist()
                if name=='raw_candidates':
                    z=int(s['ids'][j]);q.update(mask_support=int(admission['mask_support'][z]),mask_outside=int(admission['mask_outside'][z]))
            records[i]['stages'][name]=q
    for row in records:
        row['first_absent_stage']=first_absent_stage({k:q['hit'] for k,q in row['stages'].items()})
    np.savez_compressed(out/'rays.npz',**arrays)
    # Local native geometry panels. Each clay stage is prior/proposal-only; the
    # RGB panels remain exact previously rendered original/strict/interpolated.
    mesh_images={b:np.array(Image.open(root/'admission/rgb/current_moving'/b/'frames'/frame/'frame.png')) for b in ['baseline','strict','interpolated']}
    def crop_panel(box,path):
        x0,y0,x1,y1=box;w=x1-x0;hh=y1-y0;items=[]
        for b,im in mesh_images.items():items.append((b+' RGB',im[y0:y1,x0:x1]))
        px,py=np.meshgrid(np.arange(x0,x1),np.arange(y0,y1));lx,ly=landscape_pixel(px,py,camera['w']);qr=rays[ly,lx]
        for name in ['original','prior_full','anatomical_band','safe_band','raw_candidates','semantic_candidates','strict_final','interpolated_final']:
            s=stages[name];h=s['scene'].cast_rays(o3d.core.Tensor(qr));d=h['t_hit'].numpy();ids=h['primitive_ids'].numpy();valid=np.isfinite(d)
            im=np.full((*d.shape,3),15,np.uint8);tv=s['vertices'][s['triangles']];norm=np.cross(tv[:,1]-tv[:,0],tv[:,2]-tv[:,0]);norm/=np.linalg.norm(norm,axis=1)[:,None].clip(1e-15)
            shade=60+170*abs(norm[ids[valid]]@np.array([.3,.4,.866]));im[valid]=shade[:,None];items.append((name+' clay',im))
        d=portrait[y0:y1,x0:x1];good=d>0;im=np.zeros((*d.shape,3),np.uint8)
        if good.any():
            lo,hi=np.quantile(d[good],[.01,.99]);val=np.clip((d-lo)/max(hi-lo,1e-8),0,1);im[good]=np.c_[val[good]*255,(1-val[good])*255,np.full(good.sum(),100)].astype(np.uint8)
        items.append(('original depth',im))
        cols=3;canvas=Image.new('RGB',(cols*w,4*(hh+24)),(0,0,0));draw=ImageDraw.Draw(canvas)
        for j,(name,im) in enumerate(items):
            ox=(j%cols)*w;oy=(j//cols)*(hh+24);canvas.paste(Image.fromarray(im),(ox,oy+24));draw.text((ox+2,oy+4),name,fill='white')
            for qx,qy in xy:
                if x0<=qx<x1 and y0<=qy<y1:draw.rectangle((ox+qx-x0-2,oy+24+qy-y0-2,ox+qx-x0+2,oy+24+qy-y0+2),outline='magenta')
        canvas.save(path)
    for i,(qx,qy) in enumerate(xy):
        box=[max(0,int(qx)-80),max(0,int(qy)-65),min(1080,int(qx)+81),min(1920,int(qy)+66)]
        crop_panel(box,out/f'residual_{i+1}.png');records[i]['native_box']=box
    # This annotated region is explicitly posthoc visible-defect inspection,
    # not a replacement fit ROI and not a claim that shadow is missing skin.
    underchin=[530,1160,845,1360];crop_panel(underchin,out/'underchin_context.png')
    overview=Image.fromarray(rgb);draw=ImageDraw.Draw(overview)
    for i,(qx,qy) in enumerate(xy):
        draw.ellipse((qx-8,qy-8,qx+8,qy+8),outline='magenta',width=2);draw.text((max(0,int(qx)-70),int(qy)-25),f'{i+1}: {qx},{qy}',fill='magenta')
    draw.rectangle(underchin,outline='cyan',width=2);draw.text((530,1138),'Underchin visual context (posthoc)',fill='cyan');overview.save(out/'localized_overview.png')
    save(out/'result.json',dict(frame=frame,camera=camera,records=records,input_hashes=inputs,
        source_mesh_sha256=cq['source_mesh_sha256'],prior_sha256=sha(prior/'fit.npz'),script_sha256=sha(__file__),
        helpers={n:sha(Path(__file__).with_name(n)) for n in ['bake_joint_temporal_mesh.py','review_local_mhr_transfer.py','admit_mhr_local_patch_depth.py']},
        exact_original_native_depth_replayed=True,target_used_only_posthoc=True,geometry_modified=False,new_fit=False,
        anatomical_band_definition='all three neutral parent vertices 135<=y<=153cm',
        depth_parameter='Open3D t_hit for actual non-unit calibrated production ray; not Euclidean range',
        stage_triangle_id_definition='original/full/band IDs index their original triangles; local IDs index subdivision; raw onward IDs index proposals',
        underchin_visual_context_box=underchin,outputs={f.name:sha(f) for f in sorted(out.iterdir()) if f.is_file()}))
    print([(r['portrait_xy'],r['first_absent_stage'],r['stages']['prior_full']) for r in records],flush=True)


if __name__=='__main__':main()
