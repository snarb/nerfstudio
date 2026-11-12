"""Durable inference-only flying delivery, eight GPUs, validated cache and assembly."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assemble(out):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    path = json.loads((out/'path.json').read_text())
    records = path['records']
    assert len(records) == 84 and path['fps'] == 24
    assert [r['source_frame'] for r in records] == list(range(908,1254,5))+list(range(1254,1268))
    assert np.allclose(records[69]['pose'], records[70]['pose'], atol=1e-10)
    assert np.allclose(records[69]['intrinsics'], records[70]['intrinsics'], atol=1e-10)
    files = [out/'neural'/f'{r["index"]:03}'/'strict.png' if r['kind']=='nerf' else out/'gt'/f'{r["index"]:03}.png' for r in records]
    target = out/'Marshawn908_1267_flying_strict_BC_slowmo_24fps.mp4'
    tmp = target.with_name(target.stem+'.partial.mp4')
    video = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), 24, (1920,1080))
    assert video.isOpened()
    contact = Image.new('RGB',(1920,180*14))
    rows=[]
    try:
        for r, file in zip(records, files):
            im=cv2.imread(str(file));assert im is not None and im.shape==(1080,1920,3),str(file)
            video.write(im)
            panel=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).resize((320,180))
            ImageDraw.Draw(panel).text((8,8),f'{r["index"]} / {r["source_frame"]} / {r["kind"]}',fill='white')
            contact.paste(panel,((r['index']%6)*320,(r['index']//6)*180))
            rows.append(dict(index=r['index'],source_frame=r['source_frame'],kind=r['kind'],image_sha256=sha(file)))
    finally:
        video.release()
    check=cv2.VideoCapture(str(tmp));n=0
    while True:
        ok,im=check.read()
        if not ok:break
        assert im.shape==(1080,1920,3)
        n+=1
    fps=check.get(cv2.CAP_PROP_FPS);check.release()
    assert n==84 and abs(fps-24)<1e-6
    tmp.replace(target);contact.save(out/'all_frames_contact.jpg')
    receipt=dict(video=str(target),video_sha256=sha(target),decoded_frames=n,fps=fps,duration_seconds=n/fps,
                 width=1920,height=1080,path_sha256=sha(out/'path.json'),weights_modified=False,rows=rows)
    (out/'delivery_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('ASSEMBLED',str(target),flush=True)


def run(root,out,gpus):
    root,out=Path(root),Path(out)
    lock=(out/'delivery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    python=json.loads((root/'settings.json').read_text())['python']
    path_sha=sha(out/'path.json')
    code_sha=sha(root/'tools/flying_render.py')
    pending=queue.Queue()
    for index in range(70):
        receipt=out/'neural'/f'{index:03}'/'receipt.json'
        if receipt.exists():
            r=json.loads(receipt.read_text())
            if r.get('path_sha256')==path_sha and r.get('renderer_sha256')==code_sha and (receipt.parent/'strict.png').exists():continue
        pending.put(index)
    def worker(gpu):
        while True:
            try:index=pending.get_nowait()
            except queue.Empty:return
            command=[python,str(root/'tools/flying_render.py'),'--root',str(root),'--output',str(out),'--index',str(index)]
            for attempt in range(2):
                with (out/f'worker_{gpu}.log').open('ab') as log:
                    result=subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,timeout=600)
                if result.returncode==0:break
            else:raise RuntimeError(f'Frame {index} failed twice; inspect worker_{gpu}.log')
            print('FINISHED',index,'gpu',gpu,flush=True)
    with concurrent.futures.ThreadPoolExecutor(len(gpus)) as pool:
        list(pool.map(worker,gpus))
    subprocess.run([python,str(root/'tools/flying_gt.py'),'--output',str(out)],check=True)
    for i in range(70):
        r=json.loads((out/'neural'/f'{i:03}'/'receipt.json').read_text())
        assert r['path_sha256']==path_sha and r['renderer_sha256']==code_sha
    assemble(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--assemble-only',action='store_true');a=p.parse_args()
    if a.assemble_only:assemble(Path(a.output))
    else:run(a.root,a.output,[int(v) for v in a.gpus.split(',')])
