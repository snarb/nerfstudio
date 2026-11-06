#!/usr/bin/env python3
"""Four explicitly authorized CPU pilots; no frequency maps or training release."""
from __future__ import annotations
import argparse, concurrent.futures, fcntl, hashlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from prepare_marshawn_color_variants import linear_hd, sha, srgb, vendor, PQ, RIG, RAW
from marshawn_actor_coordinates import maps, raw_camera, raw_mask_coverage_to_HD
from marshawn_actor_confidence import native_confidence
from marshawn_radial_photometry import radial_ev

PILOTS=('AA','NE','JA','MC')


def identity(pid):
    try:
        parts=Path('/proc/%d/stat'%pid).read_text().split(') ')[1].split()
        return {'pid':pid,'start':parts[19],'boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'state':parts[0]}
    except FileNotFoundError:return None


def alive(record):
    current=identity(record['pid']) if record else None
    return bool(current and current['state']!='Z' and all(current[k]==record[k] for k in ['pid','start','boot']))


def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.%d.tmp'%os.getpid());tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def array_sha(array):
    h=hashlib.sha256();h.update(str(array.dtype).encode());h.update(json.dumps(list(array.shape)).encode());h.update(np.ascontiguousarray(array).tobytes());return h.hexdigest()


def exact_arrays(actual, expected):
    if set(actual)!=set(expected):raise ValueError('Semantic keys differ')
    checks={}
    for name,a in actual.items():
        b=expected[name]
        if a.shape!=b.shape or a.dtype!=b.dtype or not np.array_equal(a,b):
            detail=int(np.count_nonzero(a!=b)) if a.shape==b.shape else None
            raise ValueError('Exact semantic replay failed '+name+' mismatched='+str(detail))
        checks[name]={'sha256':array_sha(a),'shape':list(a.shape),'dtype':str(a.dtype),'exact':True}
    return checks


def scaled_camera(row, scale):
    out=dict(row)
    for key in ['w','h','fl_x','fl_y','cx','cy']:out[key]=row[key]*scale
    return out


def corrected(linear, row, contract, fit):
    ev=next(r['scalar_correction_EV'] for r in fit['camera_rows'] if r['camera']==row['physical_camera'])
    base=(linear*contract['global_base_gain']).astype(np.float32)
    field=radial_ev(row,fit['k'])-fit['radial_global_center_EV']+ev
    return srgb((base.astype(np.float64)*np.exp2(field[...,None])).astype(np.float32))


def full(crop, box):
    out=np.zeros((4320,8192),np.float32);x0,y0,x1,y1=box
    if crop.shape!=(y1-y0,x1-x0):raise ValueError('Native crop placement differs')
    out[y0:y1,x0:x1]=crop;return out


def semantics_from_projected(masks, hard, weights, veto, valid, scale):
    nominal=hard>=.5
    masks={k:v.copy()&~veto&valid for k,v in masks.items()}
    extra=nominal&~(masks['black']|masks['grey'])&~veto&valid
    db=distance_transform_edt(~masks['black']);dg=distance_transform_edt(~masks['grey'])
    masks['black']|=extra&(db<=dg);masks['grey']|=extra&(dg<db)
    overlap=masks['black']&masks['grey'];masks['black'][overlap]=False;masks['grey'][overlap]=False
    kernels=[2*r*scale+1 for r in [8,16,32]]
    core=binary_fill_holes(cv2.morphologyEx(nominal.astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernels[0],kernels[0]))))&valid&~veto
    outer=cv2.dilate(core.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernels[1],kernels[1])))>0
    yy=np.where(core)[0]
    if len(yy):
        wide=cv2.dilate(core.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernels[2],kernels[2])))>0
        cut=int(yy.min()+.75*(yy.max()-yy.min()+1));outer[:cut]|=wide[:cut]
    band=outer&~core&valid&~veto
    weights={k:v.copy() for k,v in weights.items()}
    for w in weights.values():w[~valid|veto]=0
    return dict(**masks,**weights,core=core,band=band,valid=valid,nominal=nominal&valid&~veto)


def rebuild_semantics(raw, row, scale):
    box=raw['crop_xyxy'];veto=raw_mask_coverage_to_HD(full(raw['source_ineligible'],box),row)>1e-6
    weights={};native_stats={}
    for label,width in [('hard',0),('W32',32),('W64',64)]:
        native,stats=native_confidence(raw['nominal'],raw['source_ineligible'],width)
        projected=raw_mask_coverage_to_HD(full(native,box),row);projected[veto]=0
        weights[label]=projected;native_stats[label]=stats
    masks={actor:raw_mask_coverage_to_HD(full(raw[actor],box),row)>=.5 for actor in ['black','grey']}
    mx,my=maps(row);valid=(mx>=0)&(mx<=8191)&(my>=0)&(my<=4319)
    hard=weights.pop('hard');semantic=semantics_from_projected(masks,hard,weights,veto,valid,scale)
    return semantic,veto,native_stats


def bounds(mask, padding=24):
    y,x=np.where(mask)
    if not len(x):return None
    h,w=mask.shape
    return [max(0,int(x.min())-padding),max(0,int(y.min())-padding),min(w,int(x.max())+padding+1),min(h,int(y.max())+padding+1)]


def hair_bounds(raw_box,row):
    x0,y0,x1,y1=raw_box
    x=np.linspace(x0,x1-1,33);y=np.linspace(y0,y1-1,33)
    points=np.concatenate([np.c_[x,np.full_like(x,y0)],np.c_[x,np.full_like(x,y1-1)],np.c_[np.full_like(y,x0),y],np.c_[np.full_like(y,x1-1),y]])
    K,d=raw_camera(row['physical_camera']);dst=np.array([[row['fl_x'],0,row['cx']-.5],[0,row['fl_y'],row['cy']-.5],[0,0,1.]])
    projected=cv2.undistortPoints(points[:,None].astype(np.float64),K,d,P=dst)[:,0]
    return [max(0,int(np.floor(projected[:,0].min()))-16),max(0,int(np.floor(projected[:,1].min()))-16),min(row['w'],int(np.ceil(projected[:,0].max()))+17),min(row['h'],int(np.ceil(projected[:,1].max()))+17)]


def review_images(output, camera, rgb, semantic, veto, row, hair_review):
    directory=output/'review'/camera;directory.mkdir(parents=True,exist_ok=True)
    overlay=rgb.copy();kernel=np.ones((3,3),np.uint8)
    for mask,color in [(semantic['black'],(0,255,255)),(semantic['grey'],(255,230,0)),(semantic['core'],(0,255,0)),(veto,(255,0,255))]:
        edge=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_GRADIENT,kernel)>0;overlay[edge]=color
    union=semantic['black']|semantic['grey'];scopes={'hair_native_review_scope':hair_bounds(hair_review['crop_xyxy'],row)}
    for actor in ['black','grey']:
        mask=semantic[actor];y=np.where(mask)[0];lower=mask.copy();lower[:int(y.min()+.78*(y.max()-y.min()+1))]=False
        scopes[actor+'_feet_scope']=bounds(lower,48)
    a=cv2.dilate(semantic['black'].astype(np.uint8),np.ones((33,33),np.uint8))>0
    b=cv2.dilate(semantic['grey'].astype(np.uint8),np.ones((33,33),np.uint8))>0
    contact=(a&b)|(veto&(cv2.dilate(union.astype(np.uint8),np.ones((65,65),np.uint8))>0))
    scopes['occlusion_scope']=bounds(contact,64) or bounds(union,32)
    receipts={}
    for name,box in scopes.items():
        x0,y0,x1,y1=box
        # Save exact target pixels; no rescaling of anatomical review crops.
        src=Image.fromarray(rgb[y0:y1,x0:x1]);over=Image.fromarray(overlay[y0:y1,x0:x1]);w,h=src.size
        weight=semantic['W64'][y0:y1,x0:x1];heat=np.stack([weight*255,weight*255,weight*255],axis=-1).astype(np.uint8)
        heat[veto[y0:y1,x0:x1]]=[255,0,255]
        sheet=Image.new('RGB',(w*3,h+30),'white');sheet.paste(src,(0,30));sheet.paste(over,(w,30));sheet.paste(Image.fromarray(heat),(2*w,30))
        draw=ImageDraw.Draw(sheet);draw.text((4,5),camera+' '+name+' '+str(box)+' RGB | contours | W64 (magenta veto)',fill='black')
        path=directory/(name+'.png');sheet.save(path);receipts[name]={'path':str(path),'sha256':sha(path),'crop_xyxy':box,'target_pixel_scale':1}
    overview=Image.fromarray(overlay);overview.thumbnail((960,540));overview.save(directory/'overview.jpg',quality=95)
    receipts['overview']={'path':str(directory/'overview.jpg'),'sha256':sha(directory/'overview.jpg'),'display_thumbnail':True}
    return receipts


def one_camera(payload):
    camera,data,output,source_row,contract,fit,hair_review,input_digest=payload
    data=Path(data);output=Path(output);cv2.setNumThreads(1);started=time.time()
    try:
        row=source_row['frame'];record=source_row['confidence'];old_sem=data/'semantics'/(camera+'.npz')
        for key in ['archive','raw_archive']:
            if sha(record[key])!=record[key+'_sha256']:raise ValueError('Native/confidence source changed '+key)
        for path,h in source_row['frozen_inputs'].items():
            if sha(path)!=h:raise ValueError('Frozen camera input changed '+path)
        receipt_path=output/'receipts'/(camera+'.json')
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_text())
            if receipt['input_digest']!=input_digest:raise ValueError('Pilot source generation changed')
            if not all(sha(path)==h for path,h in receipt['output_hashes'].items()):raise ValueError('Completed pilot output changed')
            return receipt
        with np.load(record['raw_archive'],allow_pickle=False) as z:raw={k:z[k].copy() for k in z.files}
        pq=vendor();low_linear,raw_receipt=linear_hd(row,pq);low=corrected(low_linear,row,contract,fit);del low_linear
        actual=np.asarray(Image.open(data/row['file_path']).convert('RGB'))
        if not np.array_equal(low,actual):raise ValueError('Exact FHD RGB replay failed pixels='+str(int(np.count_nonzero(low!=actual))))
        low_sem,low_veto,_=rebuild_semantics(raw,row,1)
        with np.load(old_sem,allow_pickle=False) as z:expected={k:z[k].copy() for k in z.files}
        checks=exact_arrays(low_sem,expected)
        actual_valid=np.asarray(Image.open(data/row['mask_path']).convert('L'))>0
        if not np.array_equal(low_sem['valid'],actual_valid):raise ValueError('Exact FHD calibration validity replay failed')
        with np.load(record['archive'],allow_pickle=False) as z:
            confidence_checks=exact_arrays({k:low_sem[k] for k in ['W32','W64']},{k:z[k] for k in ['W32','W64']})
            if not np.array_equal(low_veto,z['source_ineligible']):raise ValueError('FHD source veto replay failed')
        atomic(output/'replay'/(camera+'.json'),{'camera':camera,'rgb_byte_exact':True,'valid_exact':True,'semantic_arrays':checks,'confidence_arrays':confidence_checks,'raw':raw_receipt})
        del low_sem,low_veto,expected,actual,low
        high_row=scaled_camera(row,2);linear,second_raw=linear_hd(high_row,pq)
        if second_raw['raw_sha256']!=raw_receipt['raw_sha256']:raise ValueError('Native TIFF changed between replay and export')
        rgb=corrected(linear,high_row,contract,fit);del linear
        semantic,veto,native_stats=rebuild_semantics(raw,high_row,2)
        if (semantic['black']&semantic['grey']).any() or (semantic['core']&semantic['band']).any():raise ValueError('Actor/core/band overlap')
        if any((semantic[k][veto|~semantic['valid']]!=0).any() for k in ['W32','W64']):raise ValueError('Vetoed nonzero confidence')
        paths=[output/high_row['file_path'],output/high_row['mask_path'],output/'semantics'/(camera+'.npz')]
        Image.fromarray(rgb).save(paths[0]);Image.fromarray(semantic['valid'].astype(np.uint8)*255).save(paths[1]);np.savez_compressed(paths[2],**semantic)
        review=review_images(output,camera,rgb,semantic,veto,high_row,hair_review)
        for path,h in source_row['frozen_inputs'].items():
            if sha(path)!=h:raise ValueError('Frozen camera input changed during export '+path)
        outputs={str(p):sha(p) for p in paths};outputs.update({v['path']:v['sha256'] for v in review.values()})
        receipt={'camera':camera,'input_digest':input_digest,'frame':high_row,'source_frame':row,'raw':raw_receipt,'output_hashes':outputs,'semantic_arrays':{k:{'sha256':array_sha(v),'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in semantic.items()},'native_confidence_stats':native_stats,'review':review,'FHD_RGB_exact':True,'FHD_semantics_exact':True,'physical_kernels':[33,65,129],'native_widths':[32,64],'elapsed_seconds':time.time()-started,'source_veto_pixels':int(veto.sum()),'manual_review_complete':False,'training_eligible':False}
        atomic(receipt_path,receipt);print(json.dumps({'camera':camera,'status':'complete','elapsed_seconds':receipt['elapsed_seconds'],'replay':'exact'}),flush=True);return receipt
    except Exception as exc:
        atomic(output/'failures'/(camera+'.json'),{'camera':camera,'error':repr(exc),'traceback':traceback.format_exc(),'time':time.time()});raise


def source_contract(data):
    meta=json.loads((data/'transforms.json').read_text());ready=json.loads((data/'ready.json').read_text())
    root=data.parent.parent;manifest_path=root/'two_actor_isolation/confidence_native_hair_v2/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if sha(manifest_path)!=ready['source_confidence_sha256'] or not manifest['native_hair66_complete']:raise ValueError('Native confidence lineage mismatch')
    if sha(Path(__file__).with_name('marshawn_actor_confidence.py'))!=manifest['helper_sha256']:raise ValueError('Native confidence helper changed')
    frame=next(r for r in meta['frames'] if r['physical_camera']=='AA');source=(data/frame['file_path']).resolve();contract_path=source.parents[1]/'color_contract.json';contract=json.loads(contract_path.read_text());fitpath=Path(contract['fit']);fit=json.loads(fitpath.read_text())
    if sha(fitpath)!=contract['fit_sha256'] or fit['uses_eval_RGB'] is not False or contract.get('radial_and_scalar_baked') is not True or fit['k']!=4 or contract['global_base_gain']!=4:raise ValueError('Frozen photometry contract mismatch')
    files=[data/'transforms.json',data/'ready.json',data/'box.json',manifest_path,Path(manifest['hair_review']),contract_path,fitpath,PQ,RIG/'cameras.txt',RIG/'images.txt']
    files += [Path(__file__).with_name(name) for name in ['build_marshawn_4k_pilots.py','prepare_marshawn_color_variants.py','marshawn_actor_coordinates.py','marshawn_actor_confidence.py','marshawn_radial_photometry.py']]
    binding={str(p):sha(p) for p in files};evals=[]
    for row in meta['frames']:
        if row['file_path'] in meta['train_filenames']:continue
        record=next(r for r in ready['records'] if r['camera']==row['physical_camera'])
        paths=[row['file_path'],row['mask_path'],'semantics/'+row['physical_camera']+'.npz']
        evals.append({'frame':row,'crops':record['crops'],'files':{rel:sha(data/rel) for rel in paths}})
    return meta,ready,manifest,contract,fit,binding,evals


def run(data,output,workers):
    if not 1<=workers<=2:raise ValueError('At most two CPU workers authorized')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'build.lock').open('a') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        meta,ready,manifest,contract,fit,binding,evals=source_contract(data)
        plan={'cameras':list(PILOTS),'workers':workers,'source_bindings':binding,'evaluation':evals,'no_gpu':True,'training_eligible':False}
        digest=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
        if (output/'plan.json').exists() and json.loads((output/'plan.json').read_text())!=plan:raise ValueError('Immutable pilot plan changed')
        atomic(output/'plan.json',plan)
        for name in ['images','masks','semantics','review','replay','receipts']:(output/name).mkdir(exist_ok=True)
        shutil.copy2(data/'box.json',output/'box.json')
        for record in evals:
            for rel,h in record['files'].items():
                shutil.copy2(data/rel,output/rel)
                if sha(output/rel)!=h:raise ValueError('Copied EVAL bytes differ')
        hair={r['camera']:r for r in json.loads(Path(manifest['hair_review']).read_text())['rows']};payloads=[]
        for camera in PILOTS:
            row=next(r for r in meta['frames'] if r['physical_camera']==camera)
            if row['file_path'] not in meta['train_filenames'] or (row['w'],row['h'])!=(1920,1012):raise ValueError('Expected authorized TRAIN FHD source')
            record=next(r for r in manifest['rows'] if r['camera']==camera)
            frozen={str(data/rel):sha(data/rel) for rel in [row['file_path'],row['mask_path'],'semantics/'+camera+'.npz']}
            payloads.append((camera,str(data),str(output),{'frame':row,'confidence':record,'frozen_inputs':frozen},contract,fit,hair[camera],digest))
        atomic(output/'progress.json',{'pid':os.getpid(),'time':time.time(),'status':'running','completed':[]})
        rows=[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(one_camera,p):p[0] for p in payloads}
            for future in concurrent.futures.as_completed(futures):
                row=future.result();rows.append(row);atomic(output/'progress.json',{'pid':os.getpid(),'time':time.time(),'status':'running','completed':[r['camera'] for r in rows]})
        for path,h in binding.items():
            if sha(path)!=h:raise ValueError('Source contract changed during export')
        for record in evals:
            for rel,h in record['files'].items():
                if sha(data/rel)!=h or sha(output/rel)!=h:raise ValueError('EVAL source/output changed')
        if sha(output/'box.json')!=sha(data/'box.json'):raise ValueError('Frozen box changed')
        frames=[next(r['frame'] for r in rows if r['camera']==c) for c in PILOTS]+[r['frame'] for r in evals]
        pilot=dict(meta);pilot['frames']=frames;pilot['train_filenames']=[r['file_path'] for r in frames if r['physical_camera'] in PILOTS]
        atomic(output/'pilot_transforms.json',pilot)
        atomic(output/'complete.json',{'status':'four_pilots_ready_for_manual_review','training_eligible':False,'maps_fitted':False,'gpu_used':False,'source_plan_sha256':sha(output/'plan.json'),'cameras':list(PILOTS),'replays_exact':True,'eval_nine_files_unchanged':True,'eval_camera_rows_and_crops':evals,'box_sha256':sha(output/'box.json'),'receipts':{r['camera']:sha(output/'receipts'/(r['camera']+'.json')) for r in rows},'manual_review_complete':False})
        atomic(output/'progress.json',{'pid':os.getpid(),'time':time.time(),'status':'complete','completed':list(PILOTS)})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['run','launch','status']);parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=2)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.action=='status':print((a.output/'progress.json').read_text() if (a.output/'progress.json').exists() else 'not started');return
    if a.action=='launch':
        with (a.output/'launcher.lock').open('a') as held:
            fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
            previous=json.loads((a.output/'launcher.json').read_text()) if (a.output/'launcher.json').exists() else {}
            if alive(previous.get('process')):print('Builder already alive');return
            with (a.output/'builder.log').open('ab') as log:
                p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run','--data',str(a.data),'--output',str(a.output),'--workers',str(a.workers)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic(a.output/'launcher.json',{'pid':p.pid,'process':identity(p.pid),'time':time.time(),'cpu_workers':a.workers});print(json.dumps({'pid':p.pid}));return
    try:run(a.data,a.output,a.workers)
    except Exception as exc:
        atomic(a.output/'progress.json',{'pid':os.getpid(),'time':time.time(),'status':'failed','error':repr(exc)});raise


if __name__=='__main__':main()
