"""Meaningful cache-index and executor-crash checks; never touches campaign jobs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from common import atomic,read


class ExecutorTests(unittest.TestCase):
    def test_agent_crash_does_not_duplicate_live_worker(self):
        with tempfile.TemporaryDirectory(prefix='marshawn_executor_test_') as directory:
            root=Path(directory);marker=root/'done.json';calls=root/'calls.txt'
            code='import time,json;from pathlib import Path;p=Path('+repr(str(calls))+');p.write_text(p.read_text()+"run\\n" if p.exists() else "run\\n");time.sleep(12);Path('+repr(str(marker))+').write_text("{}")'
            atomic(root/'queue.json',dict(jobs=[dict(id='canary',command=[sys.executable,'-c',code],completion=str(marker))]))
            command=[sys.executable,str(Path(__file__).with_name('agent.py')),'--root',str(root)]
            log=(root/'test.log').open('ab');p=subprocess.Popen(command,stdout=log,stderr=log)
            deadline=time.time()+20
            while time.time()<deadline and not calls.exists():time.sleep(.2)
            self.assertTrue(calls.exists());p.kill();p.wait()
            p=subprocess.Popen(command,stdout=log,stderr=log)
            try:
                deadline=time.time()+25
                while time.time()<deadline:
                    status=read(root/'agent_state.json')
                    if status['jobs']['canary']['status']=='complete':break
                    time.sleep(.2)
                self.assertEqual(read(root/'agent_state.json')['jobs']['canary']['status'],'complete')
                self.assertEqual(calls.read_text().count('run'),1)
            finally:p.terminate();p.wait(timeout=10);log.close()


class CacheTests(unittest.TestCase):
    def test_gpu_gather_matches_cpu_shuffled_camera_ids(self):
        import torch
        from runtime import GPUCacheGather
        if not torch.cuda.is_available():self.skipTest('CUDA qualification performed on actual worker GPU')
        images=[torch.arange(4*5*3,dtype=torch.float32).reshape(4,5,3),torch.arange(3*7*3,dtype=torch.float32).reshape(3,7,3)+1000]
        batch=dict(image=images,image_idx=torch.tensor([9,2]),mask=[torch.ones(4,5,1,dtype=torch.bool),torch.zeros(3,7,1,dtype=torch.bool)])
        indices=torch.tensor([[2,2,6],[9,0,0],[2,0,2],[9,3,4]])
        gather=GPUCacheGather();actual=gather(batch,indices)
        expected=torch.stack([images[1][2,6],images[0][0,0],images[1][0,2],images[0][3,4]])
        self.assertTrue(torch.equal(actual['image'].cpu(),expected))
        self.assertTrue(torch.equal(actual['indices'],indices))
        images[0].add_(7)
        actual=gather(batch,indices)
        self.assertTrue(torch.equal(actual['image'][1].cpu(),images[0][0,0]))
        with self.assertRaises(ValueError):gather(batch,torch.tensor([[9,4,0]]))


class ExportRecoveryTests(unittest.TestCase):
    def test_recovery_after_tensor_commit_before_manifest(self):
        import torch
        from archive import export
        with tempfile.TemporaryDirectory(prefix='marshawn_export_test_') as directory:
            root=Path(directory);checkpoint=root/'source.ckpt'
            torch.save(dict(step=12,pipeline={'field':torch.arange(17,dtype=torch.float32)},optimizers={'discard':torch.ones(32)}),checkpoint)
            selection=dict(cap=10,metrics=dict(psnr=30.,ssim=.9,lpips=.1))
            path=export(root,checkpoint,selection)
            (root/'model/manifest.json').unlink()  # Simulate interrupted receipt write.
            self.assertEqual(export(root,checkpoint,selection),path)
            self.assertTrue(read(root/'model/manifest.json')['all_pipeline_tensors_roundtrip_equal'])
            self.assertNotIn('optimizers',torch.load(path,weights_only=False))
            torch.save(dict(step=13,pipeline={'field':torch.zeros(17)}),checkpoint)
            with self.assertRaises(ValueError):export(root,checkpoint,selection)


if __name__=='__main__':unittest.main()
