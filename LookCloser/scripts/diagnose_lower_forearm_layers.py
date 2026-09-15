"""Do rejected foreground and farther PM points project inside train arm regions?

Fifteen preselected veto samples only. This is not a new geometry admission rule.
"""
import numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt
from joint_temporal_texture import read,sha,atomic_json,cameras,project
from study_confidence_depth_prior import unproject
from study_foundation_anchor_bias import sample
from study_foundation_lower_forearm import ROOT,FRAME


def run():
    diagnostic=read(ROOT/'veto_diagnosis/result.json');staged=read(ROOT/FRAME/'request.json')
    rows,_,_=cameras(FRAME);lookup={r['physical_camera']:r for r in rows};views=[]
    for pair in staged['pairs']:
        cal=np.load(Path(pair['directory'])/'calibration.npz')
        for side in ['left','right']:
            e=cal['rectified_extrinsic'] if side=='left' else np.block([
                [cal['R2']@cal['E2'][:3,:3],(cal['R2']@cal['E2'][:3,3])[:,None]],
                [np.array([[0.,0.,0.,1.]])]])
            k=cal['cropped_intrinsic'] if side=='left' else cal['P2'][:,:3]
            views.append(dict(name=pair[side],K=k,E=e,mask=cal[side+'_mask']))
    dest=ROOT/'layer_diagnosis';dest.mkdir(exist_ok=False);records=[]
    for group in diagnostic['records']:
        name=group['camera'];row=lookup[name];canvas=Image.new('RGB',(4*240,5*250));draw=ImageDraw.Draw(canvas)
        for index,event in enumerate(group['samples']):
            x,y=event['native_xy'];points=unproject(row,np.array([x,x]),np.array([y,y]),
                np.array([event['proposed_depth'],event['observed_depth']]))
            membership=[]
            for ci,view in enumerate(views):
                p=points@view['E'][:3,:3].T+view['E'][:3,3];uv=p@view['K'].T;uv=uv[:,:2]/uv[:,2:]
                interior=sample(distance_transform_edt(view['mask'].astype(bool)),uv)
                inside=(p[:,2]>0)&(sample(view['mask'].astype(float),uv)>.999)
                membership.append(dict(camera=view['name'],inside=inside.tolist(),inside_distance=interior.tolist()))
                # Show actual full train RGB; preserve near/far displacement.
                native,_=project(points,[lookup[view['name']]])
                portrait=np.column_stack((native[0,:,1],1919-native[0,:,0]))
                center=portrait.mean(0)
                half=max(120,int(np.ceil(np.abs(portrait-center).max()+12)))
                lo=np.floor(center-half).astype(int)
                source=Image.open(ROOT/FRAME/(view['name']+'.png'))
                tile=source.crop((lo[0],lo[1],lo[0]+2*half,lo[1]+2*half)).resize((240,240))
                td=ImageDraw.Draw(tile)
                for point,color in zip(portrait,['red','cyan']):
                    u,v=(point-lo)*(240/(2*half))
                    td.ellipse((u-4,v-4,u+4,v+4),outline=color,width=2)
                canvas.paste(tile,(ci*240,index*250+10))
                membership[-1].update(native_crop_side=2*half,portrait_projection=portrait.tolist())
                draw.text((ci*240+3,index*250+12),view['name']+' red=prior cyan=PM',fill='white')
            records.append(dict(query_camera=name,event_index=index,source_event=event,membership=membership,
                prior_inside_views=sum(m['inside'][0] for m in membership),farther_pm_inside_views=sum(m['inside'][1] for m in membership)))
        canvas.save(dest/(name+'.png'))
    atomic_json(dest/'result.json',dict(records=records,veto_diagnosis_sha256=sha(ROOT/'veto_diagnosis/result.json'),
        script_sha256=sha(__file__),limited_to_15_preselected_samples=True,
        training_region_membership_not_surface_ground_truth=True,geometry_admission_unchanged=True,visual_status='pending',
        panels={str(p):sha(p) for p in dest.glob('*.png')}))
    print([(r['query_camera'],r['event_index'],r['prior_inside_views'],r['farther_pm_inside_views']) for r in records],flush=True)


if __name__=='__main__':run()
