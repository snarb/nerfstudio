"""Native Log3G10/RWG preparation and train-only semantic evidence for DEC5."""
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from run_dec5_sequence import read,write
from dec5_sequence_worker import sha


def rgb_matrix(primaries):
    columns=np.array([[x/y,1,(1-x-y)/y] for x,y in primaries]).T
    white=np.array([.3127/.3290,1,(1-.3127-.3290)/.3290])
    return columns*np.linalg.solve(columns,white)[None,:]


RWG_TO_709=np.linalg.solve(rgb_matrix([(.64,.33),(.30,.60),(.15,.06)]),
    rgb_matrix([(.780308,.304253),(.121595,1.493994),(.095612,-.084589)]))


def decode_log3g10(code):
    # RED Log3G10 places linear zero at code ~.09155, not at .01.
    # Verified against the frozen project's PQ/AP1 conversion on native pixels.
    return np.where(code<0,code/15.1927-.01,
                    np.expm1(code*(np.log(10)/.224282))/155.975327-.01)


def pose(camera):
    w,x,y,z=np.asarray(camera['qvec_wxyz'],np.float64)
    rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    matrix=np.eye(4);matrix[:3,:3]=rotation.T
    matrix[:3,3]=-rotation.T@np.asarray(camera['tvec'])
    matrix[:3,1:3]*=-1
    return matrix.tolist()


def prepare(root,frame,recipe):
    cv2.setNumThreads(1);manifest=read(root/'manifest.json');rig=read(root/'rig.json')
    directory=root/'frames'/frame;data=directory/'data';(data/'images').mkdir(parents=True,exist_ok=True)
    hd=directory/'semantic_data';(hd/'images').mkdir(parents=True,exist_ok=True)
    records=[];rows=[];hdrows=[];heldout=manifest['heldout_camera']
    for name,camera in sorted(rig['cameras'].items()):
        source=Path(manifest['cache'])/frame/f'{name}_{frame}.tif'
        prefix='eval' if name==heldout else 'train';relative=f'images/frame_{prefix}_{name}.png'
        destination=data/relative;small=hd/relative;receipt=directory/'image_receipts'/(name+'.json')
        digest=sha(source);old=read(receipt,{})
        if not (destination.exists() and small.exists() and old.get('source_sha256')==digest
                and old.get('decoder_sha256')==sha(__file__) and old.get('image_sha256')==sha(destination)
                and old.get('hd_sha256')==sha(small)):
            original=cv2.imread(str(source),cv2.IMREAD_UNCHANGED)
            if original is None or original.shape!=(3072,6144,3) or original.dtype!=np.uint16:
                raise ValueError('Invalid native source '+str(source))
            image=np.empty(original.shape,np.uint8)
            for start in range(0,3072,128):
                raw=original[start:start+128,:,::-1].astype(np.float32)/65535
                linear=np.maximum(decode_log3g10(raw)@RWG_TO_709.astype(np.float32).T,0)*10.570320292648677
                value=linear/(1+linear)
                value=np.where(value<=.0031308,12.92*value,1.055*np.power(value,1/2.4)-.055)
                image[start:start+len(raw)]=np.rint(np.clip(value,0,1)*255).astype(np.uint8)[...,::-1]
            for dest,im in [(destination,image),(small,cv2.resize(image,(1920,960),interpolation=cv2.INTER_AREA))]:
                tmp=dest.with_name(dest.stem+'.tmp.png')
                if not cv2.imwrite(str(tmp),im,[cv2.IMWRITE_PNG_COMPRESSION,1]):raise IOError('Image write failed')
                tmp.replace(dest)
            write(receipt,dict(source_sha256=digest,image_sha256=sha(destination),hd_sha256=sha(small),
                source_shape=[3072,6144],decoder_sha256=sha(__file__)))
        parameters=camera['params_fx_fy_cx_cy_k1_k2_p1_p2'];fx,fy,cx,cy,*dist=parameters
        if any(abs(d)>1e-12 for d in dist):raise ValueError('Unexpected calibrated distortion')
        row=dict(file_path=relative,physical_camera=name,transform_matrix=pose(camera),
            w=6144,h=3072,fl_x=fx,fl_y=fy,cx=cx,cy=cy,k1=0,k2=0,p1=0,p2=0)
        rows.append(row);other=dict(row,w=1920,h=960)
        for key in ['fl_x','fl_y','cx','cy']:other[key]*=1920/6144
        hdrows.append(other);records.append(read(receipt))
        write(directory/'progress.json',dict(phase='prepare',time=time.time(),completed=len(rows),total=65))
    if len(rows)!=65 or sum(r['physical_camera']==heldout for r in rows)!=1:raise ValueError('65-camera rig required')
    for dest,frames in [(data,rows),(hd,hdrows)]:
        meta=dict(camera_model='OPENCV',frame_id=frame,w=frames[0]['w'],h=frames[0]['h'],frames=frames,
            train_filenames=[r['file_path'] for r in frames if r['physical_camera']!=heldout],
            val_filenames=[r['file_path'] for r in frames if r['physical_camera']==heldout],
            test_filenames=[r['file_path'] for r in frames if r['physical_camera']==heldout])
        write(dest/'transforms.json',meta)
    write(directory/'prepared.json',dict(records=records,train=64,eval=1,heldout=heldout,
        no_crop=True,training_resolution=[6144,3072],display='frozen gain 10.570320292648677 + Reinhard + sRGB',
        source='16-bit Log3G10 REDWideGamut TIFF',training_encoding='8-bit display RGB; native TIFF retained on S3'))


def semantics(root,frame,recipe):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2);sys.path.insert(0,recipe['sapiens_repo'])
    from sapiens.dense.models import init_model
    config=Path(recipe['sapiens_repo'])/'sapiens/dense/configs/seg/shutterstock_goliath/sapiens2_0.4b_seg_shutterstock_goliath-1024x768.py'
    model=init_model(config,Path(recipe['sapiens_checkpoint']),device='cuda:0')
    directory=root/'frames'/frame;data=directory/'semantic_data';meta=read(data/'transforms.json')
    output=directory/'semantics';output.mkdir(exist_ok=True);records=[]
    checkpoint_digest=sha(recipe['sapiens_checkpoint'])
    for row in meta['frames']:
        if row['file_path'] not in meta['train_filenames']:continue
        source=data/row['file_path'];destination=output/(row['physical_camera']+'.npz');receipt=destination.with_suffix('.json')
        digest=sha(source);old=read(receipt,{})
        if destination.exists() and old.get('source_sha256')==digest and old.get('checkpoint_sha256')==checkpoint_digest:
            records.append(old);continue
        image=cv2.imread(str(source));portrait=np.ascontiguousarray(np.rot90(image))
        inputs=model.data_preprocessor(model.pipeline(dict(img=portrait)))
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(inputs['inputs']);logits=F.interpolate(logits.float(),size=portrait.shape[:2],mode='bilinear',align_corners=False)
            probability=logits.softmax(1)[0,[4,3,0]].permute(1,2,0).cpu().numpy()
        probability=np.ascontiguousarray(np.rot90(probability,-1))
        if not np.isfinite(probability).all():raise ValueError('Nonfinite semantics')
        temp=destination.with_name(destination.stem+'.tmp.npz')
        np.savez_compressed(temp,probability=np.rint(probability*65535).astype(np.uint16));temp.replace(destination)
        record=dict(camera=row['physical_camera'],source_sha256=digest,checkpoint_sha256=checkpoint_digest,
            output_sha256=sha(destination),semantic_shape=list(probability.shape));write(receipt,record);records.append(record)
    write(output/'complete.json',dict(records=records,uses_validation_rgb=False,semantic_probabilities_not_alpha=True))
