"""Restartable eight-GPU inference controller; no training or model mutations."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def run(root):
    root=Path(root);out=root/'operations/flying_native_hybrid_4k_20261112'
    out.mkdir(exist_ok=True)
    lock=(out/'controller.lock').open('a');fcntl.fcntl(lock,fcntl.F_SETFD,fcntl.FD_CLOEXEC)
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    base=root/'operations/flying_union_interior_20261112'
    script=root/'tools/flying_native_hybrid.py';code_sha=sha(script)
    (out/'path.json').write_bytes((base/'path.json').read_bytes())
    for name in ['codec']:
        if not (out/name).exists():(out/name).symlink_to(base/name,target_is_directory=True)
    pending=queue.Queue()
    for i in range(70):
        folder=out/'neural'/f'{i:03}';r=folder/'receipt.json'
        if r.exists():
            receipt=json.loads(r.read_text())
            if receipt['code_sha256']==code_sha and receipt['path_sha256']==sha(base/'path.json') and receipt['output_sha256']==sha(folder/'strict.png'):continue
        pending.put(i)
    def worker(gpu):
        while True:
            try:i=pending.get_nowait()
            except queue.Empty:return
            for attempt in range(2):
                with (out/f'worker_{gpu}.log').open('ab') as log:
                    result=subprocess.run([sys.executable,str(script),'--root',str(root),'--index',str(i)],
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,timeout=2400)
                if result.returncode==0:break
            else:raise RuntimeError(f'Frame {i} failed; inspect worker {gpu}')
            print('FINISHED',i,'GPU',gpu,flush=True)
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        list(pool.map(worker,range(8)))
    import cv2
    from PIL import Image,ImageDraw
    (out/'gt').mkdir(exist_ok=True)
    tail=[];frames=[]
    contact=Image.new('RGB',(1920,14*180))
    for i in range(84):
        if i>=70:
            original=base/'gt'/f'{i:03}.png'
            a=out/'gt'/f'{i:03}.png'
            rgb=cv2.imread(str(original))
            cv2.imwrite(str(a),cv2.resize(rgb,(3840,2160),interpolation=cv2.INTER_LANCZOS4))
            tail.append(dict(index=i,original_sha256=sha(original),output_sha256=sha(a),
                             original_untouched=True,operation='2x Lanczos4 only; no crop, alpha or color change'))
        else:a=out/'neural'/f'{i:03}'/'strict.png'
        frames.append(dict(index=i,sha256=sha(a)))
        im=Image.open(a);im.thumbnail((320,180));ImageDraw.Draw(im).text((5,5),str(i),fill='white')
        contact.paste(im,((i%6)*320,(i//6)*180))
    contact.save(out/'all_frames_contact.jpg')
    (out/'tail_identity.json').write_text(json.dumps(tail,indent=2))
    (out/'delivery_receipt.json').write_text(json.dumps(dict(width=3840,height=2160,rows=frames),indent=2))
    sys.path.insert(0,str(root/'tools'))
    from flying_encode import encode
    encode(out,width=3840,height=2160)
    old=out/'Marshawn908_1267_flying_strict_BC_slowmo_24fps.mp4';new=out/'Marshawn_native_hybrid_flight_4K.mp4';old.replace(new)
    r=json.loads((out/'delivery_receipt.json').read_text());r['video']=str(new)
    (out/'delivery_receipt.json').write_text(json.dumps(r,indent=2))
    print('DELIVERED',new,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
