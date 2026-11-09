#!/usr/bin/env python3
"""Bounded existing-card-only shared cos^k diagnostic; never applies corrections."""
import json, hashlib, argparse
from pathlib import Path
import cv2, numpy as np
from scipy.optimize import least_squares
from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_text,read_images_text
Y=np.array([.2722287168,.6740817658,.0536895174])
R=Path('/fsx/tmp/marshawn_001660_hd')
SOURCE=R/'report/independent_card_calibration_PRECHECK/actual_subset_v3_final'
RIG=Path('/fsx/oregon/gta_marshawn_2_1_prep_20261028/root/Marshawn_2_1/golden/rig')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fit(rows, fixed=None):
    cams=sorted({r['camera']for r in rows}); y=np.array([r['y']for r in rows]);x=np.array([r['x']for r in rows]);A=np.array([[r['camera']==c for c in cams]for r in rows],float)
    if not len(rows):raise ValueError('Empty fit')
    if fixed is not None:
        offsets={c:float(np.median([r['y']-fixed*r['x']for r in rows if r['camera']==c]))for c in cams}
        return dict(k=float(fixed),offsets=offsets)
    # Unconstrained k: negative/out-of-range is evidence against the hypothesis.
    X=np.c_[A,x];init=np.linalg.lstsq(X,y,rcond=None)[0]
    result=least_squares(lambda b:X@b-y,init,loss='soft_l1',f_scale=.1,xtol=1e-11,ftol=1e-11,gtol=1e-11)
    return dict(k=float(result.x[-1]),offsets=dict(zip(cams,map(float,result.x[:-1]))),rank=int(np.linalg.matrix_rank(X)),condition=float(np.linalg.cond(X)))
def errors(rows,m):
    v=[r['y']-m['offsets'][r['camera']]-m['k']*r['x']for r in rows if r['camera']in m['offsets']]
    return dict(cards=len(v),median_abs_EV=float(np.median(np.abs(v)))if v else None,p90_abs_EV=float(np.quantile(np.abs(v),.9))if v else None,RMSE_EV=float(np.sqrt(np.mean(np.square(v))))if v else None,signed_median_EV=float(np.median(v))if v else None)
def main(output):
    output.mkdir(parents=True,exist_ok=False)
    measurements=json.loads((SOURCE/'measurements.json').read_text()); cameras=read_cameras_text(RIG/'cameras.txt');images=read_images_text(RIG/'images.txt');rig={q.name.split('/')[0]:cameras[q.camera_id]for q in images.values()}
    def radius(name,uv):
        fx,fy,cx,cy,*d=rig[name].params
        # Actual photometry UVs are OpenCV native-raster sample indices (not
        # COLMAP pixel centers). Convert the calibrated K to this convention.
        K=np.array([[fx,0,cx-.5],[0,fy,cy-.5],[0,0,1.]])
        xy=cv2.undistortPoints(np.array(uv,dtype=np.float64).reshape(1,1,2),K,np.array(d,dtype=np.float64)).reshape(2)
        return float(xy@xy)
    by={(r['frame'],r['point']):r for r in measurements if r['camera']=='KC'};pairs=[]
    for row in measurements:
        if row['camera']=='KC':continue
        ref=by.get((row['frame'],row['point']))
        if ref is None:continue
        rt=radius(row['camera'],row['uv']);rr=radius('KC',ref['uv']);y=float(np.log2((np.array(row['rgb'])@Y)/(np.array(ref['rgb'])@Y)))
        pairs.append(dict(camera=row['camera'],frame=row['frame'],card=row['card'],point=row['point'],material=row['material'],target_uv=row['uv'],ref_uv=ref['uv'],target_r2=rt,ref_r2=rr,x=float(-.5*np.log2((1+rt)/(1+rr))),y=y))
    groups={}
    for r in pairs:groups.setdefault((r['camera'],r['frame'],r['card'],r['material']),[]).append(r)
    cards=[]
    for (cam,frame,card,material),rows in groups.items():
        cards.append(dict(camera=cam,frame=frame,card=card,material=material,patches=len(rows),x=float(np.median([r['x']for r in rows])),y=float(np.median([r['y']for r in rows])),target_r2=float(np.median([r['target_r2']for r in rows])),ref_r2=float(np.median([r['ref_r2']for r in rows]))))
    train=[r for r in cards if r['frame']==1700 and r['material']=='paper_white'and r['card']%2==0 and r['camera']!='KA']
    models={'k0_existing_equal_card_median_gains':fit(train,0),'k4_fixed_physics':fit(train,4),'k_free_robust':fit(train)}
    splits={'1700_fit_even_paper':train,'1700_holdout_odd_paper':[r for r in cards if r['frame']==1700 and r['card']%2==1 and r['material']=='paper_white'],'1740_all_paper':[r for r in cards if r['frame']==1740 and r['material']=='paper_white'],'1740_odd_paper':[r for r in cards if r['frame']==1740 and r['card']%2==1 and r['material']=='paper_white'],'1700_odd_cardstock':[r for r in cards if r['frame']==1700 and r['card']%2==1 and r['material']=='outside_cardstock'],'1740_cardstock':[r for r in cards if r['frame']==1740 and r['material']=='outside_cardstock']}
    scores={name:{split:errors(rows,m)for split,rows in splits.items()}for name,m in models.items()}
    percam={c:{name:{split:errors([r for r in rows if r['camera']==c],m)for split,rows in splits.items()}for name,m in models.items()}for c in models['k_free_robust']['offsets']}
    rng=np.random.default_rng(20261106);ids=sorted({r['card']for r in train});boot=[]
    for _ in range(300):
        sampled=rng.choice(ids,len(ids),replace=True);rows=[r for card in sampled for r in train if r['card']==card]
        if {r['camera']for r in rows}!={r['camera']for r in train}:continue
        f=fit(rows)
        if f['rank']==len(f['offsets'])+1:boot.append(f['k'])
    lco={}
    for c in models['k_free_robust']['offsets']:
        m=fit([r for r in train if r['camera']!=c]);held=[r for r in train if r['camera']==c];offset=float(np.median([r['y']-m['k']*r['x']for r in held]));m['offsets'][c]=offset
        lco[c]=dict(k_fitted_without_camera=m['k'],nuisance_offset_from_same_declared_fit_cards_only=offset,holdout_1700=errors([r for r in splits['1700_holdout_odd_paper']if r['camera']==c],m),holdout_1740=errors([r for r in splits['1740_all_paper']if r['camera']==c],m))
    ka=[]
    old=json.loads((R/'color_variants/gain4_scalar_trainfit_poseeval/complete.json').read_text());oldrows=next(v for v in old.values()if isinstance(v,list)and v and isinstance(v[0],dict)and 'exposure_correction_ev'in v[0]);ev={r['camera']:r['exposure_correction_ev']for r in oldrows}
    old_y=ev['KC']-ev['KA']
    for r in cards:
        if r['camera']=='KA':ka.append(dict(**r,existing_1660_posegain_implied_logratio=old_y,residual_to_old_offset_EV=r['y']-old_y,predicted_radial_delta_k4=4*r['x'],predicted_radial_delta_free=models['k_free_robust']['k']*r['x'],inferred_local_offset_k4=r['y']-4*r['x'],inferred_local_offset_free=r['y']-models['k_free_robust']['k']*r['x']))
    coverage={c:dict(fit_cards=len([r for r in train if r['camera']==c]),fit_card_ids=[r['card']for r in train if r['camera']==c],fit_x_range=[min(r['x']for r in train if r['camera']==c),max(r['x']for r in train if r['camera']==c)],all_target_r2_range=[min(r['target_r2']for r in pairs if r['camera']==c),max(r['target_r2']for r in pairs if r['camera']==c)])for c in models['k_free_robust']['offsets']}
    optics={c:dict(fx=float(rig[c].params[0]),fy=float(rig[c].params[1]),width=rig[c].width,height=rig[c].height,model=rig[c].model)for c in ['KC','KA','KB','LB','JA','LA']}
    report=dict(kind='SHARED_COSK_EXISTING_STATIC_CARDS_DIAGNOSTIC_NOT_APPLIED',models=models,scores=scores,per_camera_scores=percam,coverage=coverage,optics=optics,bootstrap=dict(unit='physical reference card jointly across cameras;300draws fixedseed',valid=len(boot),k_quantiles_025_50_975=np.quantile(boot,[.025,.5,.975]).tolist()),leave_camera_out_slope=lco,KA_DIAGNOSTIC_ONLY=ka,definition='log2(Ytarget/YKC)=offset_camera+k*x; x=-.5log2((1+rTarget²)/(1+rKC²)); normalized rays from native raw UV via exact OPENCV lens undistortion',fit='1700 previously EVEN physical paper cards only, equal card weight; FREE unconstrained shared k robust softL1 scale0.1EV predeclared;fixed0/4 nuisance offsets equal-card medians',expected_physical_k_range=[0,6],free_k_in_range=0<=models['k_free_robust']['k']<=6,limitations=['Bootstrap measures these cards only, not calibrated lens confidence.','View-dependent paper/cardstock BRDF and local lighting remain confounds.','Absolute unseen-camera offset cannot be inferred by leave-camera-out; slope transfer refits held-camera nuisance offset using ONLY original even fit cards.','KA has one partial odd edge card, excluded ALL model fits; no global KA correction can be accepted.','One shared k is a restricted hypothesis; lens identity/aperture response is not inferred from focal calibration alone.'],uses_target1660_eval_RGB=False,corrections_applied=False,NN_updates=0,bindings={str(p):sha(p)for p in [SOURCE/'measurements.json',SOURCE/'conclusion.json',RIG/'cameras.txt',RIG/'images.txt',Path(__file__)]})
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n');(output/'card_level.json').write_text(json.dumps(cards,indent=2)+'\n');(output/'paired_patch_radii.json').write_text(json.dumps(pairs,indent=2)+'\n')
    print(json.dumps(dict(models=models,scores=scores,bootstrap=report['bootstrap'],KA=ka,lco=lco),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
