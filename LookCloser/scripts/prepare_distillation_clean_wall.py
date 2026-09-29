"""Photographic repeating brick backdrop on the calibrated physical wall plane.

This is an explicitly simplified background prior, not a reconstruction of the
original stands, cables, shadows or the exact full-wall albedo. A clean brick
course from the independent-time photograph supplies the texture. The fitted
plane determines its depth and parallax; the actor remains unchanged.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
from mesh_distillation_background import read,write,sha


class PhotographicBrickWall(torch.nn.Module):
    """Fixed physical plane and view-independent, periodic photographic texture."""
    def __init__(self,asset):
        super().__init__();self.asset=Path(asset);self.receipt=read(self.asset/'receipt.json');g=read(self.asset/'geometry.json')
        if self.receipt['actual_eval_rgb_used'] or self.receipt['geometry_sha256']!=sha(self.asset/'geometry.json') or self.receipt['texture_sha256']!=sha(self.asset/'brick.png'):
            raise ValueError('Wall asset provenance mismatch')
        for k in ['plane','basis','origin_uv','brick_size']:self.register_buffer(k,torch.tensor(g[k],dtype=torch.float32))
        if not torch.all(self.brick_size>0) or not torch.isfinite(self.plane).all():raise ValueError('Invalid wall geometry')
        torch.testing.assert_close(self.basis@self.basis.T,torch.eye(2),rtol=0,atol=1e-5)
        torch.testing.assert_close(self.basis@self.plane[:3],torch.zeros(2),rtol=0,atol=1e-5)
        rgb=np.asarray(Image.open(self.asset/'brick.png')).astype('float32')/255
        self.register_buffer('texture',torch.tensor(rgb.transpose(2,0,1)[None].copy()));self.stagger=g['stagger']

    def forward(self,origins,directions):
        with torch.autocast(device_type=origins.device.type,enabled=False):
            origins,directions=origins.float(),directions.float();denom=(directions*self.plane[:3]).sum(-1)
            t=-((origins*self.plane[:3]).sum(-1)+self.plane[3])/torch.where(denom.abs()>1e-8,denom,torch.ones_like(denom))
            point=origins+t[...,None]*directions;uv=(point@self.basis.T-self.origin_uv)/self.brick_size
            uv[...,0]+=torch.remainder(torch.floor(uv[...,1]),2)*self.stagger
            grid=2*torch.remainder(uv,1)-1
            rgb=F.grid_sample(self.texture,grid.reshape(1,-1,1,2),align_corners=False,padding_mode='border')[0,:,:,0].T.reshape(*origins.shape[:-1],3)
            return dict(rgb=rgb,depth=t[...,None],valid=((t>0)&(denom.abs()>1e-8))[...,None])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['source','camera','plane','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(2)
    camera=read(a.camera);plane=np.array(read(a.plane)['plane']);plane/=np.linalg.norm(plane[:3]);pose=np.array(camera['transform_matrix'])
    receipt=read(a.source/'input.json');path=a.source/'source_000899.png';rgb=np.array(Image.open(path)).astype('float32')/255
    if receipt['frame']!='000899' or receipt['actual_eval_rgb_used'] or sha(path)!=receipt['generated_rgb_sha256']:raise ValueError('Independent source provenance mismatch')
    def point(x,y):
        # Explicit portrait pixel centers -> original landscape pixel centers.
        px=camera['w']-y-.5;py=x+.5
        direction=np.array([(px-camera['cx'])/camera['fl_x'],-(py-camera['cy'])/camera['fl_y'],-1.])@pose[:3,:3].T
        t=-(pose[:3,3]@plane[:3]+plane[3])/(direction@plane[:3]);return pose[:3,3]+t*direction
    # Two visible horizontal grout-center anchors in the reviewed upper-wall crop.
    left=point(150,382);right=point(1000,365);u=right-left;u/=np.linalg.norm(u)
    v=np.cross(plane[:3],u);v/=np.linalg.norm(v)
    if (point(150,420)-left)@v<0:v=-v
    height=abs((point(150,83)-left)@v)
    width=abs((point(1060,220)-point(95,220))@u)
    origin=left+u*((point(95,220)-left)@u)-v*height
    tw=1024;th=int(round(tw*height/width))
    yy,xx=np.meshgrid((np.arange(th)+.5)/th,(np.arange(tw)+.5)/tw,indexing='ij')
    xyz=origin+xx[...,None]*width*u+yy[...,None]*height*v;q=(xyz-pose[:3,3])@pose[:3,:3];z=-q[...,2]
    sx=camera['fl_x']*q[...,0]/z+camera['cx']-.5;sy=-camera['fl_y']*q[...,1]/z+camera['cy']-.5
    if np.any(z<=0) or sx.min()<1490 or sy.min()<-10 or sy.max()>1089:raise ValueError('Brick patch leaves the reviewed static upper-wall region')
    tile=cv2.remap(rgb,sx.astype('float32'),sy.astype('float32'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    # Preserve photographed interior variation; make the periodic grout boundary
    # continuous. This is an authored backdrop prior, not a depth observation.
    grout=np.median(np.concatenate([tile[:5].reshape(-1,3),tile[-5:].reshape(-1,3)]),axis=0)
    dx=np.minimum(xx,1-xx)*tw;dy=np.minimum(yy,1-yy)*th
    edge=np.exp(-.5*(np.minimum(dx,dy)/5.)**2)[...,None]
    tile=tile*(1-edge)+grout*edge
    tile=cv2.GaussianBlur(tile,(0,0),.8,borderType=cv2.BORDER_REFLECT)
    Image.fromarray(np.rint(tile.clip(0,1)*255).astype('uint8')).save(a.output/'brick.png')
    preview=np.tile(tile,(5,4,1))
    for row in range(5):
        if row%2:preview[row*th:(row+1)*th]=np.roll(preview[row*th:(row+1)*th],tw//2,axis=1)
    im=Image.fromarray(np.rint(preview.clip(0,1)*255).astype('uint8'));im.thumbnail((1200,1000));im.save(a.output/'texture_preview.jpg')
    source=Image.fromarray(np.rot90(np.rint(rgb*255).astype('uint8')));draw=ImageDraw.Draw(source)
    corners=list(zip(sy[[0,0,-1,-1],[0,-1,-1,0]].tolist(),(camera['w']-1-sx[[0,0,-1,-1],[0,-1,-1,0]]).tolist()))
    draw.line(corners+[corners[0]],fill=(0,255,255),width=4);source.crop((0,0,1080,550)).save(a.output/'source_patch_review.jpg')
    geometry=dict(plane=plane.tolist(),basis=[u.tolist(),v.tolist()],origin_uv=[float(origin@u),float(origin@v)],brick_size=[float(width),float(height)],stagger=.5)
    write(a.output/'geometry.json',geometry);(a.output/'source.py').write_text(Path(__file__).read_text())
    write(a.output/'receipt.json',dict(background='User-authorized simplified brick wall; periodic photographic patch; equipment and original cast shadows omitted',
        source_rgb_sha256=sha(path),camera_sha256=sha(a.camera),plane_sha256=sha(a.plane),geometry_sha256=sha(a.output/'geometry.json'),texture_sha256=sha(a.output/'brick.png'),
        source_frame='000899',actual_eval_rgb_used=False,physical_reserved_camera_seen=True,actor_unchanged=True,
        brick_pixel_dimensions=[tw,th],source_native_bounds=[float(sx.min()),float(sy.min()),float(sx.max()),float(sy.max())],script_sha256=sha(__file__),visual_review_pending=True))
    print(geometry,flush=True)


if __name__=='__main__':main()
