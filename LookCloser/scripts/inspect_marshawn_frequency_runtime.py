"""CPU-only host/library fingerprint for frozen frequency fits; no CUDA context."""
import argparse,hashlib,importlib.metadata,importlib.util,json,os,pathlib,subprocess,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch


def main(output):
    if output.exists():raise ValueError('Preserve runtime snapshot')
    packages={}
    for distribution,module in [('torch','torch'),('torchvision','torchvision'),('numpy','numpy'),('pytorch-msssim','pytorch_msssim'),('tinycudann','tinycudann')]:
        spec=importlib.util.find_spec(module)
        try:version=importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:version=None
        paths=[]
        if spec and spec.origin:
            parent=pathlib.Path(spec.origin).parent
            paths=list(parent.glob('*.py'))if module in ('tinycudann','pytorch_msssim')else[pathlib.Path(spec.origin)]
        packages[module]=dict(version=version,present=spec is not None,files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths})
    assert packages['pytorch_msssim']['present']and packages['tinycudann']['present']
    assert not torch.cuda.is_initialized()
    GPUs=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True).splitlines()
    output.write_text(json.dumps(dict(python=sys.executable,torch=torch.__version__,CUDA_build=torch.version.cuda,CUDA_initialized=False,packages=packages,GPUs=[dict(index=int(r.split(',')[0]),uuid=r.split(',')[1].strip(),name=r.split(',')[2].strip(),memory_MiB=int(r.split(',')[3]))for r in GPUs],per_fit_identity_join='Controller receipt + frozen worker launch records physical UUID and absolute Python command; this immutable host snapshot supplies package versions.'),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);main(p.parse_args().output)
