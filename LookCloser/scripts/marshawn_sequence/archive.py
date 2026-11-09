"""Export exact selected field tensors; optionally relocate a completed frame."""
import os
from pathlib import Path
import shutil
import subprocess
import shlex
import torch
from common import read,atomic,sha


def export(frame_root,checkpoint,selection):
    frame_root=Path(frame_root);checkpoint=Path(checkpoint)
    folder=frame_root/'model';folder.mkdir(exist_ok=True)
    path=folder/'inference.ckpt'
    if not path.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        payload=dict(step=state['step'],pipeline=state['pipeline'],inference_only=True,source_training_checkpoint_sha256=sha(checkpoint))
        tmp=path.with_suffix('.tmp');torch.save(payload,tmp)
        with tmp.open('rb') as stream:os.fsync(stream.fileno())
        # Round-trip every field/grid tensor before releasing optimizer checkpoints.
        restored=torch.load(tmp,map_location='cpu',weights_only=False)
        if restored['pipeline'].keys()!=state['pipeline'].keys():raise ValueError('Export state keys differ')
        for key,value in state['pipeline'].items():
            other=restored['pipeline'][key]
            if torch.is_tensor(value):
                if value.dtype!=other.dtype or value.shape!=other.shape or not torch.equal(value,other):raise ValueError('Export tensor differs '+key)
        os.replace(tmp,path)
    # Recover the narrow crash window between tensor-file rename and manifest.
    restored=torch.load(path,map_location='cpu',weights_only=False)
    if restored['source_training_checkpoint_sha256']!=sha(checkpoint):raise ValueError('Existing export belongs to a different training checkpoint')
    atomic(folder/'manifest.json',dict(checkpoint='inference.ckpt',sha256=sha(path),step=restored['step'],inference_cap=selection['cap'],metrics={k:selection['metrics'][k] for k in ('psnr','ssim','lpips')},optimizer_removed=True,all_pipeline_tensors_roundtrip_equal=True,source_training_checkpoint_sha256=restored['source_training_checkpoint_sha256']))
    return path


def relocate(root,frame,completion,archive_definition):
    from campaign import Host
    root=Path(root);frame_root=root/'frames'/f'{frame:06d}';host=Host(archive_definition)
    dest=host.root/f'{frame:06d}'
    # Convert only symlinks whose actual targets are inside this owned frame.
    for path in frame_root.rglob('*'):
        if not path.is_symlink():continue
        target=path.resolve()
        try:target.relative_to(frame_root.resolve())
        except ValueError:continue
        path.unlink();path.symlink_to(os.path.relpath(target,path.parent),target_is_directory=target.is_dir())
    host.command('mkdir -p '+shlex.quote(str(dest)))
    process=subprocess.Popen(['tar','-cf','-','-C',str(frame_root),'.'],stdout=subprocess.PIPE)
    receiver=subprocess.run(host.argv('tar -C '+shlex.quote(str(dest))+' -xf -'),stdin=process.stdout)
    process.stdout.close();code=process.wait()
    if code or receiver.returncode:raise RuntimeError('Completed frame archive transfer failed')
    remote_model=dest/'model/inference.ckpt'
    digest=host.command('sha256sum '+shlex.quote(str(remote_model))).split()[0]
    if digest!=completion['checkpoint_sha256']:raise ValueError('Archived model checksum differs')
    result=dict(completion,checkpoint=str(remote_model),artifact_root=str(dest),archive_ssh=archive_definition.get('ssh'))
    host.put(dest/'frame_complete.json',result)
    atomic(frame_root/'archive_receipt.json',result)
    for name in ('native','data','data_model','data_resolution','data_exr','train','model'):
        path=frame_root/name
        if path.is_dir() and not path.is_symlink():shutil.rmtree(path)
    return result
