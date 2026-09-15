"""Disjoint pure-ending work with frozen renderer and no-overwrite publication."""
import argparse
import ctypes
import errno
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime,timezone
from PIL import Image
import render_cinematic_6k_output as render

PRIMARY=render.OUT
WORK=Path('/mnt/data/dec5_cinematic_wide_spiral_6k_output_parallel_endings_v1')


def verify(folder,config):
    receipt=render.read(folder/'complete.json')
    assert receipt['request_sha256']==render.sha(PRIMARY/'request.json')
    for name,digest in receipt['hashes'].items():assert render.sha(folder/name)==digest
    result=render.read(folder/'result.json')
    assert 126<=result['index']<=148 and result['kind']=='real_train_rgb'
    assert result['train_alpha']==1 and result['output_dimensions']==[3456,6144]
    assert result['camera']==config['inventory'][result['index']]['camera']
    with Image.open(folder/'frame.png') as im:assert im.size==(3456,6144)
    provenance=render.read(folder/'source_provenance.json')
    assert provenance['worker_sha256']==config['script_sha256']
    assert provenance['decoder_sha256']==config['decoder_sha256']
    assert len(provenance['sources'])==1 and provenance['sources'][0]['camera']=='H004_C005_1210SZ'
    # Retained files must not contain references that break after moving folders.
    for path in folder.glob('*.json'):assert str(WORK) not in path.read_text()
    return result


def publish_no_replace(source,dest):
    libc=ctypes.CDLL(None,use_errno=True)
    rename=libc.renameat2
    rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    rename.restype=ctypes.c_int
    if rename(-100,os.fsencode(source),-100,os.fsencode(dest),1)!=0:
        error=ctypes.get_errno()
        raise OSError(error,os.strerror(error),str(dest))


def publish_symlink_no_replace(payload,dest):
    """Atomic directory visibility; EEXIST includes an existing empty directory."""
    os.symlink(os.path.relpath(payload,dest.parent),dest,target_is_directory=True)


def publish_internal(source,dest,config):
    namespace=PRIMARY/'parallel_completed';namespace.mkdir(exist_ok=True)
    lock=namespace/'.publication.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    try:
        payload=namespace/source.name
        if not payload.exists():
            # This namespace is exclusively ours under the atomic lock, never
            # the primary renderer's frames/ namespace. Ordinary move is safe.
            os.rename(source,payload)
        verify(payload,config)
        publish_symlink_no_replace(payload,dest)
        verify(dest,config)
        assert dest.resolve()==payload.resolve() and not os.path.isabs(os.readlink(dest))
    finally:lock.unlink()


def run(count):
    assert 1<=count<=23
    config=render.read(PRIMARY/'request.json');q=render.read(render.BASE/'request.json')
    assert render.sha(render.__file__)==config['script_sha256']
    assert render.sha(render.BASE/'request.json')==config['parent_request_sha256']
    decoder=Path(render.__file__).with_name('convert_dec5_5a3_pq16_to_exr.py')
    assert render.sha(decoder)==config['decoder_sha256']
    for name,digest in config['dependencies'].items():assert render.sha(Path(render.__file__).with_name(name))==digest
    command=['ssh',render.REMOTE,'sha256sum',render.REMOTE_ROOT+'/render_cinematic_6k_output.py',
        render.REMOTE_ROOT+'/convert_dec5_5a3_pq16_to_exr.py']
    remote=subprocess.check_output(command,text=True).splitlines()
    assert [line.split()[0] for line in remote]==[config['script_sha256'],config['decoder_sha256']]
    WORK.mkdir(exist_ok=True)
    if (WORK/'request.json').exists():assert (WORK/'request.json').read_bytes()==(PRIMARY/'request.json').read_bytes()
    else:(WORK/'request.json').write_bytes((PRIMARY/'request.json').read_bytes())
    request=dict(primary=str(PRIMARY),isolated_root=str(WORK),request_sha256=render.sha(PRIMARY/'request.json'),
        renderer_sha256=config['script_sha256'],wrapper_sha256=render.sha(__file__),
        indices=list(range(126,149)),initialization_skipped=True,remote_common_code_copied=False,
        publication='Atomic relative directory symlink to internal parallel_completed payload',
        payload_namespace=str(PRIMARY/'parallel_completed'),retained_absolute_references_checked=True,
        failed_renameat2_wrapper_sha256=render.sha(WORK/'failed_renameat2_wrapper.py'))
    assert request['failed_renameat2_wrapper_sha256']==render.read(WORK/'parallel_request.json')['wrapper_sha256']
    if (WORK/'parallel_request_symlink.json').exists():assert render.read(WORK/'parallel_request_symlink.json')==request
    else:render.write(WORK/'parallel_request_symlink.json',request)
    render.OUT=WORK
    published=0;start=time.monotonic()
    for record in q['inventory'][126:126+count]:
        frame=record['frame_id'];dest=PRIMARY/'frames'/frame
        if (dest/'complete.json').exists():verify(dest,config);continue
        primary=render.read(PRIMARY/'progress.json')
        assert primary.get('index',999)<126,'Primary reached ending jobs; preserve disjoint ownership'
        assert not dest.exists(),'Primary owns this frame directory already'
        assert render.sha(render.__file__)==config['script_sha256']
        source=WORK/'frames'/frame;payload=PRIMARY/'parallel_completed'/frame
        if payload.exists():result=verify(payload,config)
        else:render.render_frame(q,record);result=verify(source,config)
        publish_internal(source,dest,config);published+=1
        check=dict(utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),frame=frame,
            seconds=result['seconds'],primary_progress=render.read(PRIMARY/'progress.json'),
            published_directory=str(dest),complete_sha256=render.sha(dest/'complete.json'))
        with (WORK/'publication_checks.jsonl').open('a') as f:f.write(json.dumps(check)+'\n')
        print(f'published={frame} seconds={result["seconds"]:.1f} primary={check["primary_progress"].get("frame")}',flush=True)
    render.write(WORK/'progress.json',dict(stage='requested_parallel_frames_finished',pid=os.getpid(),
        requested_count=count,published_this_run=published,seconds=time.monotonic()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--count',type=int,default=2);a=p.parse_args();run(a.count)
