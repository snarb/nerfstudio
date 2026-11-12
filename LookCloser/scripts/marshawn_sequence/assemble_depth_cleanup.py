"""Assemble and verify the72-frame cleanup videos and before/after review sheets."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root, suffix=''):
    root = Path(root)
    frames = list(range(908,1268,5))
    receipt = dict(frames=72, fps=24, seconds=3, camera='GC', training_restarted=False,
                   SAM_output_masks=False, weights_modified=False, videos={}, rows=[])
    writers = {}
    paths = {}
    for name, size in [('depth_soft',(1920,1080)),('depth_strict',(1920,1080)),('comparison',(1920,540))]:
        path = root/f'Marshawn908_1263_GC_24fps_{name}{suffix}.mp4'
        writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),24,size)
        assert writer.isOpened()
        writers[name] = writer; paths[name] = path
    contact = []
    try:
        for i, frame in enumerate(frames):
            folder = root/f'{frame:06}'
            filename=lambda name:name+(suffix if name!='raw' else '')
            images = {name:cv2.imread(str(folder/(filename(name)+'.png'))) for name in ('raw','depth_soft','depth_strict')}
            assert all(im is not None and im.shape==(1080,1920,3) for im in images.values()),frame
            receipts = {name:json.loads((folder/(filename(name)+'_receipt.json')).read_text()) for name in images}
            assert len({r['checkpoint_sha256'] for r in receipts.values()}) == 1
            for name in ('depth_soft','depth_strict'):
                writers[name].write(images[name])
            panels = [cv2.resize(images[name],(960,540)) for name in ('raw','depth_strict')]
            for panel, label in zip(panels,('Before','Depth + opacity cleanup')):
                cv2.putText(panel,label,(20,32),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),1)
            writers['comparison'].write(np.concatenate(panels,axis=1))
            if i%6==0 or frame in (923,988,1188,1263):
                p=cv2.resize(images['depth_strict'],(480,270))
                cv2.putText(p,str(frame),(10,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                contact.append(p)
            receipt['rows'].append(dict(frame=frame,model_sha256=receipts['raw']['checkpoint_sha256'],
                soft_png_sha256=digest(folder/(filename('depth_soft')+'.png')),strict_png_sha256=digest(folder/(filename('depth_strict')+'.png'))))
    finally:
        for writer in writers.values():writer.release()
    for name,path in paths.items():
        capture = cv2.VideoCapture(str(path))
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT))==72
        assert capture.get(cv2.CAP_PROP_FPS)==24
        decoded=0
        while capture.read()[0]:decoded+=1
        assert decoded==72
        capture.release()
        receipt['videos'][name]=dict(path=str(path),sha256=digest(path),size_bytes=path.stat().st_size)
    while len(contact)%4:contact.append(np.zeros_like(contact[0]))
    cv2.imwrite(str(root/('strict_contact'+suffix+'.jpg')),np.concatenate([np.concatenate(contact[i:i+4],axis=1) for i in range(0,len(contact),4)],axis=0))
    (root/('delivery_receipt'+suffix+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt['videos'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--suffix',default='')
    a=p.parse_args();main(a.output,a.suffix)
