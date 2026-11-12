"""Exact calibrated hemisphere spline ending in a native BC photographic crop."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline,PchipInterpolator
from scipy.spatial.transform import Rotation,RotationSpline
from scipy.signal import savgol_filter


def smooth(t):
    t=np.clip(t,0,1);return t*t*t*(10+t*(-15+6*t))


def full_intrinsics(c):
    r=c['view'];x,y,_,_=c['crop']
    return np.array([r['fl_x'],r['fl_y'],r['cx']+x,r['cy']+y])


def points(root,inventory,frame):
    z=np.load(root/'operations/depth_cleanup_20261112'/f'{frame:06}'/'depth_strict_v3.npz')
    a=z['accumulation'][::6,::6,0];d=z['depth'][::6,::6,0]
    ys,xs=np.mgrid[:z['depth'].shape[0]:6,:z['depth'].shape[1]:6]
    gc=inventory['GC'];K=full_intrinsics(gc)*1920/json.loads((root/'original_camera_sizes.json').read_text())['GC'][0]
    rays=np.stack([(xs+.5-K[2])/K[0],-(ys+.5-K[3])/K[1],-np.ones_like(xs)],axis=-1)
    rays/=np.linalg.norm(rays,axis=-1,keepdims=True);p=np.array(gc['pose'])
    return (p[:3,3]+(rays@p[:3,:3].T)*d[...,None])[a>.8]


def crop_at(index):
    #8% animated crop zoom, then full native1:1 at the end; no source pre-resize.
    u=float(smooth((index-70)/13));width=round(2074+(1920-2074)*u)
    height=round(width*9/16);cx=3725.;cy=1440-140*u
    return [round(cx-width/2),round(cy-height/2),width,height]


def build(root,out):
    root,out=Path(root),Path(out)
    inventory={r['name']:r for r in json.loads((out/'camera_inventory.json').read_text())['cameras']}
    names=['GC','FB','EB','DD','CD','BC'];knots=np.array([0,.20,.40,.70,.96,1.])
    control=np.array([inventory[n]['pose'] for n in names]);xyz=control[:,:3,3]
    ring=np.array([c['position'] for n,c in inventory.items() if n.endswith('C')])
    circle=np.linalg.lstsq(np.c_[2*ring[:,:2],np.ones(len(ring))],(ring[:,:2]**2).sum(1),rcond=None)[0]
    center=np.array([circle[0],circle[1],.04]);delta=xyz-center
    spherical=np.stack([np.unwrap(np.arctan2(delta[:,1],delta[:,0])),np.arctan2(delta[:,2],np.linalg.norm(delta[:,:2],axis=1)),np.linalg.norm(delta,axis=1)],axis=1)
    trajectory=PchipInterpolator(knots,spherical)
    rotation=RotationSpline(knots,Rotation.from_matrix(control[:,:3,:3]))
    focals=CubicSpline(knots,np.log([full_intrinsics(inventory[n])[:2] for n in names]))
    last=inventory['BC'];lastK=full_intrinsics(last);endcrop=crop_at(70);bc=np.array(last['pose'])
    # Match the photographic endpoint exactly, including its off-axis principal point.
    K_end=np.array([lastK[0]*1920/endcrop[2],lastK[1]*1080/endcrop[3],(lastK[2]-endcrop[0])*1920/endcrop[2],(lastK[3]-endcrop[1])*1080/endcrop[3]])
    cloud=points(root,inventory,1253);local=(cloud-bc[:3,3])@bc[:3,:3]
    uv=np.c_[lastK[0]*local[:,0]/-local[:,2]+lastK[2],-lastK[1]*local[:,1]/-local[:,2]+lastK[3]]
    midpoint=np.array(endcrop[:2])+np.array(endcrop[2:])/2
    nearby=np.argsort(((uv-midpoint)**2).sum(1))[:100]
    ray=np.array([(midpoint[0]-lastK[2])/lastK[0],-(midpoint[1]-lastK[3])/lastK[1],-1.]);ray/=np.linalg.norm(ray)
    target=bc[:3,3]+(bc[:3,:3]@ray)*np.median(np.linalg.norm(cloud[nearby]-bc[:3,3],axis=1))
    records=[];autofit=[]
    for i in range(70):
        q=i/69;u=.5-.5*np.cos(np.pi*q);az,el,r=trajectory(u)
        pose=np.eye(4);pose[:3,3]=center+r*np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)]);pose[:3,:3]=rotation(u).as_matrix()
        if i==69:pose=bc.copy()
        f=np.exp(focals(u));cloud=points(root,inventory,908+5*i);local=(cloud-pose[:3,3])@pose[:3,:3];local=local[local[:,2]<-.05]
        uv=np.c_[f[0]*local[:,0]/-local[:,2],-f[1]*local[:,1]/-local[:,2]]
        low,high=np.quantile(uv,[.002,.998],axis=0);scale=min(1920*.65/(high[0]-low[0]),1080*.65/(high[1]-low[1]))
        autofit.append([np.log(scale),*(low+high)/2])
        records.append(dict(index=i,source_frame=908+5*i,kind='nerf',pose=pose.tolist(),focal=f.tolist(),spline_u=float(u)))
    auto=savgol_filter(np.array(autofit),11,3,axis=0)
    for i,row in enumerate(records):
        f=np.array(row.pop('focal'));pose=np.array(row['pose']);q=i/69;weight=float(smooth((q-.48)/.52));scale=np.exp(auto[i,0])
        K=np.r_[f*scale,[960,540]-scale*auto[i,1:]]
        local=(target-pose[:3,3])@pose[:3,:3]
        targetK=np.r_[f*1920/endcrop[2], [960,540]-np.array([f[0]*local[0]/-local[2],-f[1]*local[1]/-local[2]])*1920/endcrop[2]]
        K=(1-weight)*K+weight*targetK
        if i==69:K=K_end.copy()
        row['intrinsics']=K.tolist();row['width']=1920;row['height']=1080
    # Same actual train pose at the splice; source frames1254..1267 displayed24fps=>5x slow.
    for i,frame in enumerate(range(1254,1268),start=70):
        crop=crop_at(i);sx,sy=1920/crop[2],1080/crop[3]
        K=[lastK[0]*sx,lastK[1]*sy,(lastK[2]-crop[0])*sx,(lastK[3]-crop[1])*sy]
        records.append(dict(index=i,source_frame=frame,kind='native_gt',pose=bc.tolist(),intrinsics=K,width=1920,height=1080,native_crop=crop))
    manifest=dict(fps=24,source_fps=120,duration_seconds=3.5,frames=84,slowmo_start_index=70,slowmo_seconds=14/24,slowmo_speed=.2,
        endpoint_camera='BC',controls=names,spherical_center=center.tolist(),azimuth_arc_degrees=float(abs(spherical[-1,0]-spherical[0,0])*180/np.pi),
        source_pre_resize=False,final_crop_sampling='selected native crop to fixed FullHD in one pass; final crop1920x1080 is1:1',
        notes='All source action908..1267 covered;70 ordinary24fps samples then14 original120fps images. Camera pose equalsBC atindex69 and remains fixed.',records=records)
    (out/'path.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k!='records'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();build(a.root,a.output)
