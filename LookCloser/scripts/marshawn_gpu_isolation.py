"""Fail-closed physical GPU UUID isolation for single-device experiment workers."""
import os,re

def require_environment(expected_uuid):
 if not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',expected_uuid):raise ValueError('Physical NVIDIA GPU UUID required')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=expected_uuid:raise RuntimeError('CUDA_VISIBLE_DEVICES must equal the exact leased physical UUID before Torch import')
 return dict(CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],expected_physical_uuid=expected_uuid)

def child_environment(expected_uuid):
 environment=dict(os.environ,CUDA_VISIBLE_DEVICES=expected_uuid)
 if not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',expected_uuid):raise ValueError('Physical NVIDIA GPU UUID required')
 return environment

def verify_cuda(expected_uuid,torch_module):
 proof=require_environment(expected_uuid)
 count=torch_module.cuda.device_count()
 if count!=1:raise RuntimeError('Exactly one visible CUDA device required')
 current=torch_module.cuda.current_device()
 if current!=0:raise RuntimeError('Single-device worker must use logical CUDA0')
 properties=torch_module.cuda.get_device_properties(0)
 if not hasattr(properties,'uuid'):raise RuntimeError('Runtime cannot prove actual physical CUDA UUID')
 actual=str(properties.uuid)
 if not actual.startswith('GPU-'):actual='GPU-'+actual
 if actual.lower()!=expected_uuid.lower():raise RuntimeError('CUDA properties physical UUID differs from acquired lease')
 return dict(**proof,visible_device_count=count,logical_current_device=current,actual_physical_uuid=actual,name=properties.name)
