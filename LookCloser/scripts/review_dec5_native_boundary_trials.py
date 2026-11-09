#!/usr/bin/env python3
"""Finish all matched native contour trials with sealed ROI metrics and crops."""
from pathlib import Path
import argparse,subprocess,sys,time
import cv2,numpy as np
from run_dec5_sequence import read,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    scripts=Path(__file__).parent;arms=['control','uniform','frequency'];output=r/'native_boundary_review';output.mkdir(exist_ok=True)
    while not all((r/('native_boundary_'+arm)/'complete.json').exists() for arm in arms):time.sleep(20)
    results={};tiles=[];meta=read(r/'native_frame901/data/transforms.json')
    f=next(f for f in meta['frames'] if f['file_path'] in meta['val_filenames'])
    gt=cv2.imread(str(r/'native_frame901/data'/f['file_path']))[...,::-1].astype(np.float32)/255
    probability=np.load(r/'eval_reference_hd.npz')['probability'];domain=(probability[...,0]>.5*65535)|(probability[...,1]>.5*65535)
    yy,xx=np.nonzero(domain);s=gt.shape[1]/domain.shape[1]
    y0,y1=max(0,int((yy.min()-30)*s)),min(gt.shape[0],int((yy.max()+31)*s))
    x0,x1=max(0,int((xx.min()-30)*s)),min(gt.shape[1],int((xx.max()+31)*s))
    def tile(label,image):
        crop=np.rot90(np.uint8(np.clip(image[y0:y1,x0:x1],0,1)*255+.5)[...,::-1])
        crop=cv2.resize(crop,(320,round(crop.shape[0]*320/crop.shape[1])),interpolation=cv2.INTER_AREA)
        cv2.putText(crop,label,(8,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(30,220,255),2);return crop
    tiles.append(tile('GT',gt))
    for arm in arms:
        directory=r/('native_boundary_'+arm);training=directory/'frames/000901'/('training_native_boundary_'+arm)
        with (output/(arm+'_score.log')).open('w') as log:
            subprocess.run([sys.executable,str(scripts/'score_dec5_sequence_pilot.py'),'--frame-root',str(directory/'frames/000901'),
                '--training',str(training),'--reference',str(r/'eval_reference_hd.npz'),'--output',str(output/(arm+'.json')),
                '--include-person-roi','--resize-evaluation-reference'],stdout=log,stderr=subprocess.STDOUT,check=True)
        results[arm]=read(output/(arm+'.json'))['rows'][-1]
        history=read(training/'history.json');pred=np.load(Path(history[-1]['renders'])/'0.npz')['rgb']
        tiles.append(tile(arm,pred));del pred
    changes={}
    for arm in arms[1:]:
        changes[arm]={region:100*(value['lpips']/results['control']['regions'][region]['lpips']-1)
                      for region,value in results[arm]['regions'].items()}
    cv2.imwrite(str(output/'comparison.jpg'),np.concatenate(tiles,axis=1))
    write(output/'complete.json',dict(results=results,lpips_percent_change=changes,updates=5000,
        native_resolution=[6144,3072],reference_resampled_for_metrics_only=True,
        production_recipe_changed=False,physical_time_flicker_measured=False))


if __name__=='__main__':main()
