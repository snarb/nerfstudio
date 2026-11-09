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


class SequenceAllocationTests(unittest.TestCase):
    def test_split_source_ranges_still_balance_both_workstations(self):
        from types import SimpleNamespace
        from campaign import allocate_frames
        rows=[dict(name='ws07',production_weight=1.4,frame_range=[908,1111]),dict(name='ws08',production_weight=1.4,frame_range=[1112,1268])]+[dict(name='silver'+str(i),production_weight=1.) for i in range(8)]+[dict(name=n,production_weight=.8) for n in ('olive','clever')]
        allocation=allocate_frames([SimpleNamespace(row=row) for row in rows])
        self.assertEqual(sorted(f for frames in allocation.values() for f in frames),list(range(908,1269)))
        self.assertEqual(len(allocation[0]),41);self.assertEqual(len(allocation[1]),41)
        self.assertTrue(all(908<=f<=1111 for f in allocation[0]))
        self.assertTrue(all(1112<=f<=1268 for f in allocation[1]))


class TrustedBackgroundTests(unittest.TestCase):
    def test_hair_uncertainty_other_people_and_sensor_edges_are_protected(self):
        import numpy as np
        from worker_background import trusted_background
        actor=np.zeros((500,500),bool);actor[250,250]=True
        valid=np.ones_like(actor);valid[100,103]=False
        alpha=np.zeros(actor.shape,np.uint8);alpha[250,250]=255;alpha[100,102]=1
        unknown=np.zeros_like(actor);unknown[100,101]=True
        mask=trusted_background(dict(black=actor,grey=np.zeros_like(actor),valid=valid,soft_alpha=alpha),unknown)
        self.assertTrue(mask[100,100]);self.assertFalse(mask[250,250])
        self.assertFalse(mask[186,186]);self.assertFalse(mask[100,101])
        self.assertFalse(mask[100,102]);self.assertFalse(mask[100,103])
        self.assertFalse(mask[63,100]);self.assertFalse(mask[450,100])


if __name__=='__main__':unittest.main()
