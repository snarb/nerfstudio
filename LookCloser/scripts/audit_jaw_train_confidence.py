"""Validate both confidence arms and matched production-wrapper canaries."""
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
from joint_temporal_texture import read,sha,atomic_json
from study_jaw_boundary_notches import PARENT
from study_jaw_depth_footprint import BASE
from study_confidence_depth_prior import load_real


def audit(root):
    parent=read(PARENT/'request.json');records=[];pairs=[]
    spots=read('/mnt/data/dec5_elevated_camera_jaw_review_150/end_diagnosis/spot_audit.json')['selected_components']
    for frame in ['001193','001195']:
        rows,depths,receipt=load_real(BASE/'analysis',frame)
        if receipt!=read(BASE/'analysis'/frame/'request.json')['real_depth_receipt']:raise ValueError('Changed native depths')
        del depths
        source=next(r for r in parent['inventory'] if r['frame_id']==frame)
        if sha(source['mesh'])!=source['mesh_sha256']:raise ValueError('Original mesh changed')
        old=o3d.io.read_triangle_mesh(source['mesh']);ov,ot=np.asarray(old.vertices),np.asarray(old.triangles)
        _,oldcc,_=old.cluster_connected_triangles()
        for arm in ['train_anchor','footprint']:
            folder=root/arm/'guarded'/frame;g=read(folder/'result.json');gr=read(folder/'request.json')
            analysis=root/arm/'analysis'/frame;a=read(analysis/'result.json');ar=read(analysis/'request.json')
            if g['request_sha256']!=sha(folder/'request.json') or g['mesh_sha256']!=sha(folder/'mesh.ply'):raise ValueError('Changed guard')
            if gr['diagnosis_sha256']!=sha(analysis/'result.json') or a['evidence_sha256']!=sha(analysis/'evidence.npz') or a['request_sha256']!=sha(analysis/'request.json'):raise ValueError('Changed confidence inputs')
            if ar['real_depth_receipt']!=receipt or g['depth_receipt']!=receipt:raise ValueError('Different measured depths')
            if not g['observed_free_space_guard_passed'] or len(g['rounds'][-1]['checks'])!=124 or any(c['trusted_free_pixels'] for c in g['rounds'][-1]['checks']):raise ValueError('Native ray veto failed')
            mesh=o3d.io.read_triangle_mesh(str(folder/'mesh.ply'));v,t=np.asarray(mesh.vertices),np.asarray(mesh.triangles)
            if not np.array_equal(v,ov) or not np.array_equal(t[:len(ot)],ot):raise ValueError('Changed original geometry')
            labels,cc,_=mesh.cluster_connected_triangles();labels=np.asarray(labels)
            islands=set(labels[len(ot):])-set(labels[:len(ot)])
            if islands or len(cc)>len(oldcc):raise ValueError('New disconnected additions')
            edges=np.sort(t[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1);_,counts=np.unique(edges,axis=0,return_counts=True)
            if (counts>2).any():raise ValueError('Nonmanifold edges')
            records.append(dict(frame=frame,arm=arm,added_triangles=g['final_added'],components=len(cc),
                new_islands=len(islands),guard_result_sha256=sha(folder/'result.json'),moving=g['moving']))
        rootrgb=root/'footprint';folder=rootrgb/'rgb_review'/frame;crop_panels=[]
        x0,y0,x1,y1=next(s for s in spots if s['frame_id']==frame)['bbox_inclusive']
        for kind,names in [('rgb',['old_moving','phase_moving']),('train_rgb',['F004_E005_1210FP','M004_B005_12109O'])]:
            for name in names:
                renders=[];images=[];depthmaps=[]
                for variant in ['baseline','guarded']:
                    out=rootrgb/kind/frame/name/variant;dest=out/'frames'/frame;r=read(dest/'complete.json')
                    if r['request_sha256']!=sha(out/'request.json'):raise ValueError('Changed render request')
                    for p,h in r['hashes'].items():
                        if sha(dest/p)!=h:raise ValueError('Changed render output')
                    res=read(dest/'result.json');renders.append(res)
                    if len(res['source_cameras'])!=62 or 'F004_B005_1210O9' in res['source_cameras'] or res['rgb_averaging']:raise ValueError('Renderer protocol changed')
                    image=np.asarray(Image.open(dest/'frame.png').convert('RGB'));images.append(image)
                    d=np.rot90(np.load(dest/'target_depth.npz')['depth']);depthmaps.append(d)
                    if image.shape!=(1920,1080,3) or not np.isfinite(d).all():raise ValueError('Invalid render')
                if renders[0]['camera']!=renders[1]['camera'] or renders[0]['fixed_exposure']!=renders[1]['fixed_exposure']:raise ValueError('Unmatched cameras/exposure')
                row=dict(frame=frame,kind=kind,camera=name,changed_rgb_pixels=int(np.any(images[0]!=images[1],axis=2).sum()))
                if name=='old_moving':
                    row['spot_rgb_black']=[int((im[y0:y1+1,x0:x1+1].max(2)==0).sum()) for im in images]
                    row['spot_depth_misses']=[int((d[y0:y1+1,x0:x1+1]<=0).sum()) for d in depthmaps]
                    crop_panels=[Image.fromarray(im).crop((x0-65,y0-65,x1+66,y1+66)) for im in images]
                pairs.append(row)
        w,h=crop_panels[0].size;panel=Image.new('RGB',(2*w,h+25));draw=ImageDraw.Draw(panel)
        for i,image in enumerate(crop_panels):panel.paste(image,(i*w,25));draw.text((i*w+3,5),['baseline','train + footprint'][i],fill='white')
        panel.save(folder/'spot_native.png')
    for arm in ['train_anchor','footprint']:
        if read(root/arm/'guarded/001193/request.json')['rule']!=read(root/arm/'guarded/001195/request.json')['rule']:raise ValueError('Per-frame rule exception')
    atomic_json(root/'audit.json',dict(script_sha256=sha(__file__),records=records,render_pairs=pairs,
        heldout_metrics_computed=False,no_full_frame_quality_metrics=True,changed_pixel_counts_are_localization_diagnostics=True,
        production_replaced=False))
    print(records,pairs,flush=True)
    if (root/'visual_review.json').exists():
        review=read(root/'visual_review.json');images={p:sha(root/p) for p in review['inspected_images']}
        atomic_json(root/'artifact_manifest.json',dict(retained_hashes={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p!=root/'artifact_manifest.json'},reviewed_image_hashes=images,
            status='positive_local_canary_full_goal_incomplete'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/mnt/data/dec5_jaw_train_confidence'))
    audit(p.parse_args().root)
