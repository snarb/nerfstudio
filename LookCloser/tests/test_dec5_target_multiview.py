from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from refine_dec5_target_multiview import patch_consensus,choose_candidates,sample
from dec5_hair import project,unproject


def test_train_view_patch_consensus_recovers_known_plane_without_target_rgb():
    target=dict(h=64,w=64,fl_x=100.,fl_y=100.,cx=32.,cy=32.,transform_matrix=np.diag([1.,-1.,-1.,1.]).tolist())
    yy,xx=np.indices((64,64));pixels=np.stack((xx+.5,yy+.5),-1)
    centers=np.stack(np.meshgrid(np.arange(24.5,40.,3),np.arange(24.5,40.,3)),axis=-1).reshape(-1,2)
    depths=np.broadcast_to(np.array([1.8,2.,2.2]),(len(centers),3))
    patch=np.array([(x,y) for y in [-1,0,1] for x in [-1,0,1]])
    points=unproject(centers[:,None,None,:]+patch[None,None],np.broadcast_to(depths[...,None],depths.shape+(9,)),target)
    observations=[]
    for offset in [-.3,-.1,.1,.3]:
        source=dict(target);pose=np.array(target['transform_matrix']);pose[0,3]=offset;source['transform_matrix']=pose.tolist()
        world=unproject(pixels,np.full((64,64),2.),source)
        texture=(.4+.15*np.sin(70*world[...,0])+.15*np.sin(65*world[...,1]+np.sin(30*world[...,0]))).astype(np.float32)
        coords,_=project(points,source);observations.append(sample(texture,coords))
    cost,support=patch_consensus(np.stack(observations),np.ones((4,len(centers),3),bool))
    assert (cost.argmin(-1)==1).mean()>.9
    assert np.all(support==4)


def test_flat_or_ambiguous_photometry_and_worse_orientation_keep_baseline():
    patches=np.ones((4,2,3,9),np.float32)
    photo,count=patch_consensus(patches,np.ones((4,2,3),bool))
    assert np.isinf(photo).all() and (count==0).all()
    best,good=choose_candidates(np.array([[1.,.1,.11],[1.,.1,.8],[1.,.1,.8]]),
        np.array([[.02,.01,.01],[.02,.03,.04],[.02,.01,.01]]),np.full((3,3),4))
    assert good.tolist()==[False,False,True] and best[2]==1
