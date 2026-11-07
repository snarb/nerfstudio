"""Publish verified central preparation to a training host before GPU dispatch.

rsync dereferences prepared RGB symlinks. No native masks, model weights or
learned occupancy are copied into another frame. The central dispatch marker
is committed only after remote SHA verification of every immutable input.
"""
from pathlib import Path
import argparse, hashlib, json, shlex, subprocess
from marshawn_video_train import sha,read,atomic,owned


def ready_inputs(ready):
    return {k:v for k,v in ready.items() if k not in ('inventory','box_sha256','frequency_policy','recipe_source')}


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def inventory(data,frame):
    ready=read(data/'ready.json');prepared=read(data/'prepared_ready.json');tf=read(data/'transforms.json')
    if ready['frame']!=frame or prepared['frame']!=frame or ready['train']!=66:raise ValueError('Wrong prepared frame/ALL66')
    if len(ready['records'])!=66 or len(tf['train_filenames'])!=66:raise ValueError('Incomplete cameras')
    paths={'prepared_ready.json','transforms.json','box.json'}
    admission=data/'admission_inputs/manifest.json'
    if admission.exists():
        if prepared.get('admission_inputs_manifest_sha256')!=sha(admission):raise ValueError('Admission input manifest changed')
        paths.add('admission_inputs/manifest.json')
        for row in read(admission)['records']:
            if sha(owned(data,data/row['archive']))!=row['archive_sha256']:raise ValueError('Admission input archive changed')
            paths.add(row['archive'])
    rows={r['camera']:r for r in ready['records']}
    for f in tf['frames']:
        camera=f['physical_camera'];row=rows[camera]
        if sha(data/row['image'])!=row['rgb_sha256'] or sha(data/'semantics'/(camera+'.npz'))!=row['semantic_sha256']:
            raise ValueError('Prepared RGB/semantics changed: '+camera)
        paths.update([row['image'],f['mask_path'],'semantics/'+camera+'.npz'])
    return dict(schema=1,frame=frame,train=66,ready_inputs_sha256=digest(ready_inputs(ready)),
        files_sha256={p:sha(data/p) for p in sorted(paths)})


def verify(data,frame):
    data=Path(data);manifest=read(data/'staged_input_manifest.json');actual=inventory(data,frame)
    if actual!=manifest:raise ValueError('Staged immutable input inventory differs')
    return dict(frame=frame,verified=True,manifest_sha256=sha(data/'staged_input_manifest.json'),
                input_files=len(actual['files_sha256']))


def push(source_campaign,frame,target,target_root,target_python,host_key):
    source=Path(source_campaign).resolve();frame_dir=source/'frames'/('%06d'%frame);data=frame_dir/'data'
    marker=frame_dir/'dispatch_ready.json'
    remote_root=Path(target_root);remote_data=remote_root/'frames'/('%06d'%frame)/'data'
    manifest=inventory(data,frame)
    existing=read(marker) if marker.exists() else None
    if existing and (existing['host']!=host_key or existing['input_manifest_sha256']!=digest(manifest)):
        raise ValueError('Published training inputs changed or moved hosts')
    remote_script=remote_root/'code/LookCloser/scripts/marshawn_video_stage.py'
    check=[target_python,str(remote_script),'verify','--data',str(remote_data),'--frame',str(frame)]
    def remote(argv):
        return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',target,shlex.join(argv)],
                              capture_output=True,text=True,check=True,timeout=120)
    if not existing:
        atomic(data/'staged_input_manifest.json',manifest)
        remote(['mkdir','-p',str(remote_data)])
        # No --delete: same-frame frequency fitting may already have published
        # source-bound maps. A duplicate publication never deletes remote state.
        subprocess.run(['rsync','-rLt','--safe-links','--exclude=admission/',
                        '--exclude=*.partial*','--exclude=*.fit.lock',str(data)+'/',target+':'+str(remote_data)+'/'],
                       check=True,timeout=3600)
    verified=json.loads(remote(check).stdout)
    if verified['manifest_sha256']!=sha(data/'staged_input_manifest.json'):raise ValueError('Remote publication differs')
    receipt=dict(schema=1,frame=frame,host=host_key,remote_root=str(remote_root),
        input_manifest_sha256=digest(manifest),remote_verification=verified,
        no_model_or_learned_state_staged=True)
    atomic(marker,receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['verify','push'])
    p.add_argument('--frame',type=int,required=True);p.add_argument('--data',type=Path)
    p.add_argument('--source-campaign',type=Path);p.add_argument('--target');p.add_argument('--target-root')
    p.add_argument('--target-python');p.add_argument('--host-key')
    a=p.parse_args()
    result=verify(a.data,a.frame) if a.command=='verify' else push(a.source_campaign,a.frame,a.target,a.target_root,a.target_python,a.host_key)
    print(json.dumps(result))
