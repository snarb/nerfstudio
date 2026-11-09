"""Install one isolated campaign root and its durable local GPU executors."""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from common import atomic,read


def setup(a):
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    settings=dict(python=a.python,sam_python=a.python,tools=str(root/'tools'),source_code=str(root/'runtime_base'),frozen_helpers=str(root/'frozen_helpers'),nerfacc_extension=a.extension,dependency_path=a.dependencies,gpu_cache=False)
    if a.sam_root:
        sam=Path(a.sam_root);settings.update(sam_dependency_path=str(sam/'src')+':'+str(sam/'deps')+':'+a.dependencies,sam_checkpoint=str(sam/'weights/sam3.pt'))
    if a.memmap:settings.update(sampling_memmap_root=a.memmap,gpu_cache=True)
    if a.owners:settings['source_owners']=read(a.owners)
    atomic(root/'settings.json',settings)
    template=root/'variants/uniform16';ready=read(template/'ready.json')
    shutil.copy2(template/'box.json',root/'box.json');atomic(root/'split.json',dict(heldout=ready['validation']))
    native=root/'data/000908';native.mkdir(parents=True,exist_ok=True)
    for name in ('images','masks','parts','semantics','heads','cores'):
        if not (native/name).exists():(native/name).symlink_to((template/name).resolve(),target_is_directory=True)
    for name in ('transforms.json','box.json'):shutil.copy2(template/name,native/name)
    atomic(native/'native_ready.json',ready)
    if a.install_services:
        import getpass
        user=a.service_user or getpass.getuser()
        unit='''[Unit]\nDescription=Marshawn908 GPU executor %%i\nAfter=network-online.target\n\n[Service]\nType=simple\nUser=%s\nWorkingDirectory=%s\nEnvironment=PYTHONUNBUFFERED=1\nEnvironment=OPENBLAS_NUM_THREADS=1\nExecStart=/usr/bin/python3 %s/agent.py --root %s/agents/gpu%%i\nRestart=on-failure\nRestartSec=20\nKillMode=process\nLimitNOFILE=65536\n\n[Install]\nWantedBy=multi-user.target\n'''%(user,root,root/'tools',root)
        with tempfile.NamedTemporaryFile('w',suffix='.service',delete=False) as stream:stream.write(unit);path=stream.name
        subprocess.run(['sudo','-n','install','-m','644',path,'/etc/systemd/system/marshawn908-agent@.service'],check=True)
        Path(path).unlink();subprocess.run(['sudo','-n','systemctl','daemon-reload'],check=True)
        for gpu in a.gpus:
            (root/'agents'/('gpu'+str(gpu))).mkdir(parents=True,exist_ok=True)
            subprocess.run(['sudo','-n','systemctl','enable','--now','marshawn908-agent@%s.service'%gpu],check=True)
    print('HOST_SETUP_COMPLETE',root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--python',required=True);p.add_argument('--dependencies',default='');p.add_argument('--extension');p.add_argument('--sam-root');p.add_argument('--memmap');p.add_argument('--owners');p.add_argument('--service-user');p.add_argument('--gpus',type=int,nargs='+',default=[0]);p.add_argument('--install-services',action='store_true');setup(p.parse_args())
