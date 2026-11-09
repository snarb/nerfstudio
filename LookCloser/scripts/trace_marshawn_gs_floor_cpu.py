#!/usr/bin/env python3
"""CPU gsplat reference projection/SH/pixel contribution diagnosis, no training."""
import argparse,json,sys,hashlib
from pathlib import Path
import torch,numpy as np
from PIL import Image,ImageDraw

def run(a):
 from gsplat.cuda._torch_impl import _quat_scale_to_covar_preci,_fully_fused_projection,_spherical_harmonics
 torch.set_num_threads(2);a.output.mkdir(parents=True,exist_ok=False);state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['pipeline'];get=lambda k:state[next(p for p in state if p.endswith('gauss_params.'+k))].float();means=get('means');q=get('quats');scales=get('scales').exp();opacity=get('opacities').sigmoid().reshape(-1);coeff=torch.cat([get('features_dc')[:,None],get('features_rest')],1)
 with torch.no_grad():cov,_=_quat_scale_to_covar_preci(q,scales,compute_covar=True,compute_preci=False)
 geometry=np.load(a.geometry);n=np.array(geometry['normal']);d=float(geometry['offset']);height=means.numpy()@n+d;aspect=(scales.max(-1).values/scales.min(-1).values).numpy();meta=json.loads((a.data/'transforms.json').read_text());requests={'DC':[[600,800],[450,720],[700,650]],'HC':[[1050,820],[800,780],[1200,850]],'KA':[[1040,820],[1100,700],[900,700]]};reports=[]
 for name,pixels in requests.items():
  row=next(r for r in meta['frames'] if r['physical_camera']==name);pose=torch.tensor(row['transform_matrix'],dtype=torch.float32);R=pose[:3,:3]*torch.tensor([1,-1,-1]);vm=torch.eye(4);vm[:3,:3]=R.T;vm[:3,3]=-R.T@pose[:3,3];K=torch.tensor([[row['fl_x'],0,row['cx']],[0,row['fl_y'],row['cy']],[0,0,1.]])
  with torch.no_grad():radii,m2d,z,conic,_=_fully_fused_projection(means,cov,vm[None],K[None],row['w'],row['h']);color=(_spherical_harmonics(3,means-pose[:3,3],coeff)+.5).clamp_min(0)
  radii=radii[0].numpy();m2d=m2d[0].numpy();z=z[0].numpy();conic=conic[0].numpy();colors=color.numpy();dc=(.5+0.28209479177387814*get('features_dc')).numpy();op=opacity.numpy();pred=np.array(Image.open(a.predictions/(name+'_prediction.png')))/255;gt=np.array(Image.open(a.data/row['file_path']))/255
  for x,y in pixels:
   dx=m2d[:,0]-(x+.5);dy=m2d[:,1]-(y+.5);candidate=(radii>0)&(abs(dx)<=radii)&(abs(dy)<=radii);ids=np.flatnonzero(candidate);power=.5*(conic[ids,0]*dx[ids]**2+2*conic[ids,1]*dx[ids]*dy[ids]+conic[ids,2]*dy[ids]**2);alpha=np.minimum(.999,op[ids]*np.exp(-power));keep=(power>=0)&(alpha>=1/255);ids=ids[keep];alpha=alpha[keep];order=np.argsort(z[ids],kind='stable');ids=ids[order];alpha=alpha[order];trans=np.r_[1.,np.cumprod(1-alpha[:-1].astype(float))];use=trans*(1-alpha)>=1e-4;ids=ids[use];alpha=alpha[use];trans=trans[use];weights=alpha*trans;T=float(np.prod(1-alpha.astype(float)));rgb=np.clip((colors[ids]*weights[:,None]).sum(0)+T*np.array([.1490,.1647,.2157]),0,1);rank=np.argsort(weights)[::-1]
   local=np.array([(x+.5-row['cx'])/row['fl_x'],-(y+.5-row['cy'])/row['fl_y'],-1]);direction=pose[:3,:3].numpy()@local;direction/=np.linalg.norm(direction);origin=pose[:3,3].numpy();t=-(origin@n+d)/(direction@n);floor_z=t*(-pose[:3,2].numpy()@direction)
   top=[]
   for k in rank[:20]:i=ids[k];top.append(dict(id=int(i),weight=float(weights[k]),depth_z=float(z[i]),signed_floor_height=float(height[i]),means=means[i].tolist(),scale_xyz=scales[i].tolist(),aspect=float(aspect[i]),opacity=float(op[i]),view_rgb=colors[i].tolist(),DC_rgb=dc[i].tolist()))
   report=dict(camera=name,pixel_xy=[x,y],GT_rgb=gt[y,x].tolist(),native_PNG_rgb=pred[y,x].tolist(),CPU_reconstructed_rgb=rgb.tolist(),CPU_vs_native_PNG_max_abs=float(abs(rgb-pred[y,x]).max()),trace_opacity=float(weights.sum()),floor_plane_intersection_z=float(floor_z),weighted_expected_z=float((weights*z[ids]).sum()/max(weights.sum(),1e-12)),weight_below_floor_003=float(weights[height[ids]<-.03].sum()),weight_far_below_floor_01=float(weights[height[ids]<-.1].sum()),weight_above_floor_003=float(weights[height[ids]>.03].sum()),weight_near_floor_0015=float(weights[abs(height[ids])<.015].sum()),weight_aspect_over100=float(weights[aspect[ids]>100].sum()),weight_aspect_over10000=float(weights[aspect[ids]>10000].sum()),weighted_SH_minus_DC_l2=float((weights*np.linalg.norm(colors[ids]-dc[ids],axis=1)).sum()),top_contributors=top);reports.append(report)
   np.savez_compressed(a.output/(name+f'_{x}_{y}_trace.npz'),ids=ids,weights=weights,alpha=alpha,transmittance=trans,depth=z[ids],height=height[ids],aspect=aspect[ids],view_rgb=colors[ids],DC_rgb=dc[ids])
 out=dict(checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),plane_normal=n.tolist(),plane_offset=d,rows=reports,implementation='gsplat1.4.0 purePyTorchCPU reference covariance,projection,andSH; analytic depth-sorted alpha weights with nativecutoffs',limitations=['Reference is not the nativeCUDA rasterizer; compare reconstructedRGB against nativePNG per queriedpixel','Floorplane uses manualtrain-floor witnesses; mat has real thickness above floor','Below-floor Gaussians may approximate view-dependent reflections; association alone does not establish a safe deletion policy'])
 (a.output/'audit.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps([dict(camera=r['camera'],pixel=r['pixel_xy'],RGBerror=r['CPU_vs_native_PNG_max_abs'],below=r['weight_far_below_floor_01'],extreme=r['weight_aspect_over10000'],SHdiff=r['weighted_SH_minus_DC_l2']) for r in reports]))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['checkpoint','geometry','data','predictions','output']:p.add_argument('--'+k,type=Path,required=True)
 run(p.parse_args())
