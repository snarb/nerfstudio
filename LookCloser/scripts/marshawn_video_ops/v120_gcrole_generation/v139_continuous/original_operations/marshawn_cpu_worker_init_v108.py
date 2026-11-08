"""Execution-only CPU limits qualified against immutable actual1690 inputs."""
import os
_LIMIT=None
def initialize():
 global _LIMIT
 os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
 import cv2
 cv2.setNumThreads(1)
 from threadpoolctl import threadpool_limits
 _LIMIT=threadpool_limits(limits=1)
