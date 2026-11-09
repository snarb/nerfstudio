"""Copy the first-frame experiment bundle from Silver to an authorized SSH host."""
import argparse
from pathlib import Path
import shlex
import subprocess
from common import read,atomic
from campaign import Host


def replicate(source,definition):
    src=Path(source);row=read(definition);host=Host(row);root=host.root
    settings=read(src/'settings.json')
    host.command('mkdir -p '+shlex.join([str(root/n) for n in ('variants/uniform16','runtime_base','tools','frozen_helpers','data/000908')]))
    for name in ('runtime_base','tools','frozen_helpers','variants/uniform16','data/000908/learned'):
        if not (src/name).exists():continue
        host.command('mkdir -p '+shlex.quote(str(root/name)))
        archive=subprocess.Popen(['tar','-chf','-','--exclude=linear','-C',str(src/name),'.'],stdout=subprocess.PIPE)
        result=subprocess.run(host.argv('tar -C '+shlex.quote(str(root/name))+' -xf -'),stdin=archive.stdout)
        archive.stdout.close();code=archive.wait()
        if code or result.returncode:raise RuntimeError('Dataset copy failed '+name)
    expression="import sys;from pathlib import Path;r=Path("+repr(str(root))+");sys.path.insert(0,str(r/'tools'));from common import read,atomic;s=read(r/'settings.json');s.update(model_units_scale=.2217203609534452,source_owners="+repr(settings.get('source_owners',[]))+");atomic(r/'settings.json',s);from assets import scale_variant,map_variant;scale_variant(r/'variants/uniform16',r/'variants/model_uniform16',s['model_units_scale']);[map_variant(r/'variants/model_uniform16',r/'data/000908',r/'variants'/('model_'+p),p) for p in ['ssim090','ssim095']];print('REPLICA_ASSETS_READY')"
    setup=[row['python'],str(root/'tools/setup_host.py'),'--root',str(root),'--python',row['python'],'--dependencies',row.get('dependencies','')]
    if row.get('extension'):setup+=['--extension',row['extension']]
    if row.get('sam_root'):setup+=['--sam-root',row['sam_root']]
    host.command(shlex.join(setup),timeout=120)
    host.command('PYTHONPATH='+shlex.quote(row.get('dependencies',''))+' '+shlex.quote(row['python'])+' -c '+shlex.quote(expression),timeout=180)
    # System services are installed through the SSH login's existing sudo rights.
    login=Host({k:v for k,v in row.items() if k!='run_as_user'})
    user=row.get('service_user','ubuntu')
    unit='[Unit]\nDescription=Marshawn908 GPU executor\nAfter=network-online.target\n[Service]\nType=simple\nUser='+user+'\nWorkingDirectory='+str(root)+'\nEnvironment=PYTHONUNBUFFERED=1\nEnvironment=OPENBLAS_NUM_THREADS=1\nExecStart=/usr/bin/python3 '+str(root/'tools/agent.py')+' --root '+str(root/'agents/gpu0')+'\nRestart=on-failure\nRestartSec=20\nKillMode=process\nLimitNOFILE=65536\n[Install]\nWantedBy=multi-user.target\n'
    code="from pathlib import Path;p=Path('/etc/systemd/system/marshawn908-agent.service');p.write_text("+repr(unit)+")"
    host.command('mkdir -p '+shlex.quote(str(root/'agents/gpu0')))
    login.command('sudo -n python3 -c '+shlex.quote(code))
    login.command('sudo -n systemctl daemon-reload && sudo -n systemctl enable --now marshawn908-agent.service')
    settings=host.get(root/'settings.json')
    expression="from campaign import make_spec;make_spec("+repr(str(root))+",dict(frequency_source_policy='uniform16'),"+repr(str(root/'variants/model_uniform16'))+","+repr(str(root/'technical/port_v5'))+",32)"
    host.command('PYTHONPATH='+shlex.quote(str(root/'tools'))+' '+shlex.quote(row['python'])+' -c '+shlex.quote(expression),timeout=120)
    job=dict(id='technical_port_v5',command=[row['python'],str(root/'tools/worker.py'),'--spec',str(root/'technical/port_v5/spec.json'),'--technical-steps','32'],completion=str(root/'technical/port_v5/stage_complete.json'),target=32,progress=str(root/'technical/port_v5/progress.json'),env=dict(PYTHONPATH=row.get('dependencies',''),CUDA_VISIBLE_DEVICES='0'))
    host.put(root/'agents/gpu0/queue.json',dict(jobs=[job]))
    print('REPLICA_PREFLIGHT_QUEUED',row['name'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--definition',required=True);a=p.parse_args();replicate(a.source,a.definition)
