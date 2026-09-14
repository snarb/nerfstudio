"""Point-only control: applying the shape prior before irreversible skin admission."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from joint_temporal_texture import read,sha,atomic_json,project
from study_confidence_depth_prior import unproject
from annotation_mask_domain import known_domain
import study_forearm_plane_transfer_v3 as prior


def run(root,frame):
    prior.configure();v1=prior.v2.v1;rows,_,_=v1.cameras(frame)
    ref=next(r for r in rows if r['physical_camera']==v1.NAMES[0]);masks=v1.masks(frame)
    path=prior.OUT/frame/'plane/evidence.npz';e=np.load(path);xy=e['all_candidate_xy']
    fitpath=Path('/mnt/data/dec5_forearm_multiview_anchors')/frame/'result.json'
    fit=next(r for r in read(fitpath)['fit'] if r['model']=='quadratic')
    q=(xy-fit['reference_center'])/100
    inv=np.column_stack([q,np.ones(len(q)),q[:,0]**2,q[:,0]*q[:,1],q[:,1]**2])@fit['all_camera_coefficients']
    if not np.isfinite(inv).all() or (inv<=0).any():raise ValueError('Invalid diagnostic quadric')
    z=1/inv;points=unproject(ref,xy[:,0],xy[:,1],z);support=np.zeros(len(z),int);outside=np.zeros(len(z),bool)
    for row in rows:
        name=row['physical_camera']
        if name not in masks:continue
        uv,d=project(points,[row]);uv,d=uv[0],d[0];pix=np.rint(uv).astype(int)
        valid=known_domain(uv,d,row['w'],row['h']);ids=np.flatnonzero(valid)
        inside=np.zeros(len(z),bool);inside[ids]=masks[name][pix[ids,1],pix[ids,0]]
        support+=inside;outside|=valid&~inside
    new=(support>=2)&~outside;old=e['accepted'][xy[:,1],xy[:,0]];added=new&~old
    output=root/frame;output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'point_control.npz',xy=xy,quadric_depth=z,previous=old,quadric_semantic=new)
    gtpath=prior.OUT/frame/'rgb'/(v1.NAMES[0]+'.png');im=np.array(Image.open(gtpath));im[xy[added,1],xy[added,0]]=[255,30,30]
    Image.fromarray(np.rot90(im)).crop((0,1400,430,1920)).save(output/'newly_admitted_reference_points.png')
    result=dict(frame=frame,total_points=len(z),previous=int(old.sum()),quadric_semantically_valid=int(new.sum()),
        newly_valid=int(added.sum()),lost=int((old&~new).sum()),geometry_changed=False,
        scope='Point-only diagnostic before boundary feather, extent and depth guards; not a mesh result',
        source_evidence_sha256=sha(path),fit_result_sha256=sha(fitpath),source_gt_sha256=sha(gtpath),
        helper_sha256=sha(Path(__file__).with_name('annotation_mask_domain.py')),script_sha256=sha(__file__),
        hashes={n:sha(output/n) for n in ['point_control.npz','newly_admitted_reference_points.png']},visual_status='pending')
    atomic_json(output/'result.json',result);print(frame,result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frames',nargs='+',default=['001029','001033','001037'])
    p.add_argument('--root',type=Path,default=Path('/mnt/data/dec5_forearm_admission_order_probe'));a=p.parse_args()
    for frame in a.frames:run(a.root,frame)
