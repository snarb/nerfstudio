#!/usr/bin/env python3
"""Independent AA initialization/input-stream probe: no optimizer or forward."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-sha256',required=True);p.add_argument('--preprocess-sha256',required=True)
    a=p.parse_args()
    if sha(a.image)!=a.source_sha256: raise ValueError('AA source changed')
    import numpy as np
    import torch
    import tinycudann as tcnn
    import nerfstudio.scripts.lookcloser_preprocess as prep
    if sha(inspect.getfile(prep))!=a.preprocess_sha256: raise ValueError('Frozen preprocessing source changed')
    if torch.cuda.device_count()!=1: raise ValueError('Pin one explicitly authorized GPU')
    # This probe is explicitly permitted to overlap read-only within10GiB.
    torch.cuda.set_per_process_memory_fraction((10<<30)/torch.cuda.get_device_properties(0).total_memory,0)
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    def tensor_hash(t):
        return hashlib.sha256(t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    def rng():
        return {'torch_cpu':tensor_hash(torch.get_rng_state()),'torch_cuda':tensor_hash(torch.cuda.get_rng_state()),
                'numpy':hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest()}
    seeded=rng()
    source=prep.load_image_as_tensor(a.image,torch.device('cuda'))
    if tuple(source.shape)!=(2024,3840,3): raise ValueError('Unexpected source raster')
    model=prep.InstantNGP2D(n_levels=16,n_features=2,min_res=16,max_res=8192,log2_hashmap_size=23).to('cuda')
    parameters={k:{'sha256':tensor_hash(v),'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in model.state_dict().items()}
    before_samples=rng()
    iy=torch.randint(0,2024,(16384,),device='cuda')
    ix=torch.randint(0,3840,(16384,),device='cuda')
    after_samples=rng()
    after_parameters={k:{'sha256':tensor_hash(v),'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in model.state_dict().items()}
    if parameters!=after_parameters: raise ValueError('Probe changed initial model')
    env=Path(torch.__file__).parent.parent
    bindings=list((env/'tinycudann_bindings').glob('*.so'))
    libraries=[Path(torch.__file__).parent/'_C.cpython-311-x86_64-linux-gnu.so',*bindings]
    gpu=subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=index,uuid,name,compute_cap,driver_version','--format=csv,noheader'],text=True).strip()
    receipt={'pid':os.getpid(),'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'gpu':gpu,'python':sys.version,'torch':torch.__version__,'torch_cuda_build':torch.version.cuda,'torch_git_version':torch.version.git_version,
             'torch_config':torch.__config__.show(),'library_sha256':{str(x.relative_to(env)):sha(x) for x in libraries},'tcnn_python_sha256':sha(tcnn.__file__),
             'source_sha256':a.source_sha256,'source_float_tensor_sha256':tensor_hash(source),'preprocess_sha256':a.preprocess_sha256,'probe_sha256':sha(__file__),
             'initial_parameters':parameters,'seeded_RNG':seeded,'pre_sample_RNG':before_samples,'post_sample_RNG':after_samples,
             'first_iy_sha256':tensor_hash(iy),'first_ix_sha256':tensor_hash(ix),'first_indices_sha256':tensor_hash(torch.stack([iy,ix],dim=-1)),
             'default_dtype':str(torch.get_default_dtype()),'encoding_dtype':str(getattr(model.encoding,'dtype',None)),'decoder_dtype':str(getattr(model.decoder,'dtype',None)),
             'allow_tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'allow_tf32_cudnn':torch.backends.cudnn.allow_tf32,
             'encoding_constructor':str(inspect.signature(tcnn.Encoding)),'network_constructor':str(inspect.signature(tcnn.Network)),
             'python_random_seeded':False,'optimizer_created':False,'optimizer_steps':0,'forward_calls':0,'model_unchanged':True,
             'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
             'resource_policy':'Root-authorized read-only overlap; <=10GiB allocator cap; no producer interruption'}
    if receipt['gpu_peak_allocated_bytes']>10<<30: raise ValueError('Memory budget exceeded')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'output':str(a.output),'gpu':receipt['physical_gpu'],'peak_bytes':receipt['gpu_peak_allocated_bytes'],'model_unchanged':True}))


if __name__=='__main__':main()
