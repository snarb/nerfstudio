"""CPU-only source corroboration, sharing locked caches with target ranking."""
import argparse
import os
from pathlib import Path
import cv2
from luster_hybrid_common import read
from source_surface_confidence_luster_hybrid import source_confidence


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU-only invocation required'
    os.nice(10);cv2.setNumThreads(2)
    config=read(a.config);selections=read(a.root/'preflight/sources.json')['sources']
    sources=list(dict.fromkeys(c for target in ['7','12','95','150'] for c in selections[target]))
    for source in sources:source_confidence(a.root,source,config)
