"""Native review for a fresh per-time prior and optional admitted completion.

Current moving camera comes from the actual production request. Posthoc crops
and enclosed ray-miss components never enter fitting or admission.
"""
from pathlib import Path
import argparse, importlib
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import label
from transfer_local_mhr_prior import settings,configure,verify
from run_local_mhr_completion import adapt,read,save,sha,require

VIEWS=['current_moving','F004_E','M004_B','C004_E','G004_B']


def camera_crops(spec):
    from joint_temporal_texture import cameras
    from study_confidence_depth_prior import project_integer
    s=settings(spec);rows,_,_=cameras(spec['frame']);parent=read(spec['production_request'])
    source=next(r for r in parent['inventory'] if r['frame_id']==spec['frame'])
    a=np.load(s['HEAD']/'initial.npz');take=(a['neutral'][:,1]>140)&(a['neutral'][:,1]<177)&(abs(a['neutral'][:,0])<12)
    xy,z=project_integer(source['camera'],a['vertices'][take]);xy=xy[z>0];portrait=np.c_[xy[:,1],1919-xy[:,0]]
    lo=np.floor(portrait.min(0)-35).astype(int);hi=np.ceil(portrait.max(0)+35).astype(int)
    box=[max(0,int(lo[0])),max(0,int(lo[1])),min(1080,int(hi[0])),min(1920,int(hi[1]))]
    predictions={r['camera']:r for r in read(s['RGB']/'inference.json')['records'] if r['detected']==1}
    result={}
    for view in VIEWS:
        row=source['camera'] if view=='current_moving' else next(r for r in rows if r['physical_camera'].startswith(view))
        if view=='current_moving':crop=box
        else:
            x0,y0,x1,y1=predictions[row['physical_camera']]['native_review_box'];crop=[x0,y0,x1,min(1550,y1+200)]
        result[view]=dict(camera=row,crop=crop)
    return source,result


def panel(images,path,crop):
    x0,y0,x1,y1=crop;w=x1-x0;h=y1-y0;out=Image.new('RGB',(len(images)*w,h+24));draw=ImageDraw.Draw(out)
    for j,(name,im) in enumerate(images):
        out.paste(Image.fromarray(im).crop(crop),(j*w,24));draw.text((j*w+2,4),name,fill='white')
    out.save(path)


def enclosed_misses(depth,crop):
    x0,y0,x1,y1=crop;missing=depth[y0:y1,x0:x1]<=0;l,n=label(missing)
    border=set(np.r_[l[0],l[-1],l[:,0],l[:,-1]].tolist());mask=np.zeros(depth.shape,bool)
    local=(l>0)&~np.isin(l,list(border));mask[y0:y1,x0:x1]=local
    return mask


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initial','clay','render','rgb'])
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--completion',type=Path);p.add_argument('--views',nargs='+',default=VIEWS);a=p.parse_args()
    spec=read(a.spec);verify(spec);s=settings(spec)
    if a.stage=='initial':
        configure(spec);importlib.import_module('review_mhr_local_head_prior').main(['initial']);return
    require(a.completion is not None,'Completion root required');out=a.completion/'admission'
    config=read(a.completion/'config.json');require(config['spec']['frame']==spec['frame'],'Completion frame mismatch')
    source,views=camera_crops(spec);require(sha(source['mesh'])==source['mesh_sha256'],'Production changed')
    paths={'baseline':Path(source['mesh']),**{b:out/'silhouette100'/b/'mesh.ply' for b in ['strict','interpolated']}}
    provenance=dict(frame=spec['frame'],spec_sha256=sha(a.spec),completion_config_sha256=sha(a.completion/'config.json'),
        camera_request_sha256=sha(spec['production_request']),crops=views,mesh_hashes={k:sha(v) for k,v in paths.items()},
        script_sha256=sha(__file__),target_used_posthoc_only=True,production_modified=False)
    if a.stage=='render':
        import review_mhr_production_patch_control as old
        def binding():return dict(frame=spec['frame'],production_mesh_sha256=source['mesh_sha256'],explicit_generic_binding=provenance)
        fn,proof=adapt(old,'render',[("Path('/mnt/data/dec5_elevated_camera_dynamic_150/request.json')","Path(PRODUCTION_REQUEST)",1),
            ("'old_moving'","'current_moving'",1)],dict(ROOT=a.completion,OUT=out,FRAME=spec['frame'],ARM='silhouette100',
            PARENT=Path(spec['production_request']).parent,PRODUCTION_REQUEST=spec['production_request'],binding=binding))
        save(a.completion/('render_adapter_'+'_'.join(a.views)+'.json'),dict(proof,review_binding=provenance));fn(a.views);return
    dest=a.completion/('native_clay' if a.stage=='clay' else 'native_rgb');dest.mkdir(exist_ok=True);records=[];files=[]
    if a.stage=='clay':
        import open3d as o3d
        from admit_mhr_local_patch_depth import Scene2
        from bake_joint_temporal_mesh import camera_depth
        meshes={k:o3d.io.read_triangle_mesh(str(path)) for k,path in paths.items()}
        scenes={}
        for k,m in meshes.items():m.compute_triangle_normals();scenes[k]=Scene2(np.asarray(m.vertices),np.asarray(m.triangles))
    for view in a.views:
        crop=views[view]['crop'];images=[];data={};receipt=dest/(view+'.json');require(not receipt.exists(),'Review already exists')
        for branch in paths:
            if a.stage=='clay':
                d,ids,_=camera_depth(scenes[branch],views[view]['camera']);valid=np.isfinite(d);rgb=np.full((*d.shape,3),20,np.uint8)
                rgb[valid]=(60+170*abs(np.asarray(meshes[branch].triangle_normals)[ids[valid]]@np.array([.3,.4,.866])))[:,None]
                rgb=np.rot90(rgb);depth=np.rot90(np.where(valid,d,0))
            else:
                folder=out/'rgb'/view/branch/'frames'/spec['frame'];complete=read(folder/'complete.json')
                require(complete['request_sha256']==sha(folder.parent.parent/'request.json'),'Render request changed')
                for name,h in complete['hashes'].items():require(sha(folder/name)==h,'Render output changed')
                rgb=np.array(Image.open(folder/'frame.png'));depth=np.rot90(np.load(folder/'target_depth.npz')['depth'])
            data[branch]=(rgb,depth);images.append((branch,rgb))
        original,bd=data['baseline'];holes=enclosed_misses(bd,crop);stats=[]
        for branch,(rgb,d) in data.items():
            new=(d>0)&(bd<=0);common=(d>0)&(bd>0);delta=np.zeros(d.shape);delta[common]=d[common]-bd[common]
            stats.append(dict(branch=branch,enclosed_original_misses=int(holes.sum()),enclosed_remaining=int((holes&(d<=0)).sum()),
                new_hits=int(new.sum()),new_hits_without_rgb=int((new&(rgb.max(2)==0)).sum()),
                newly_black=int(((original.max(2)>0)&(rgb.max(2)==0)).sum()),lost_hits=int(((bd>0)&(d<=0)).sum()),
                nearer_over003=int((delta<-.003).sum()),maximum_nearer=float(max(0,-delta.min())),farther_over003=int((delta>.003).sum())))
            if a.stage=='rgb' and branch!='baseline':
                for kind,mask in [('nearer',delta<-.003),('newly_black',(original.max(2)>0)&(rgb.max(2)==0)),('untextured',new&(rgb.max(2)==0))]:
                    lab,count=label(mask);components=sorted(range(1,count+1),key=lambda i:int((lab==i).sum()),reverse=True)[:5]
                    for rank,index in enumerate(components):
                        y,x=np.nonzero(lab==index);box=[max(0,int(x.min())-25),max(0,int(y.min())-25),min(1080,int(x.max())+26),min(1920,int(y.max())+26)]
                        marked=rgb.copy();marked[lab==index]=[255,0,255];path=dest/f'{view}_{branch}_{kind}_{rank+1}.png'
                        panel([('baseline',original),(branch,rgb),('marked',marked)],path,box);files.append(dict(path=str(path),sha256=sha(path)))
        path=dest/(view+'.png');panel(images,path,crop);files.append(dict(path=str(path),sha256=sha(path)))
        record=dict(provenance,view=view,statistics=stats,enclosed_misses_are_not_confirmed_anatomical_holes=True,files=files.copy())
        save(receipt,record);records.append(record);print(view,stats,flush=True)


if __name__=='__main__':main()
