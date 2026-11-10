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
    def test_late_child_completion_reconciles_failed_parent(self):
        with tempfile.TemporaryDirectory(prefix='marshawn_late_receipt_') as directory:
            root=Path(directory);marker=root/'done.json'
            atomic(root/'queue.json',dict(jobs=[dict(id='late',command=[sys.executable,'-c','raise RuntimeError("must not relaunch")'],completion=str(marker),max_attempts=1)]))
            atomic(root/'agent_state.json',dict(jobs={'late':dict(status='failed',attempts=1,job=read(root/'queue.json')['jobs'][0])}))
            atomic(marker,{})
            log=(root/'test.log').open('ab');p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('agent.py')),'--root',str(root)],stdout=log,stderr=log)
            try:
                deadline=time.time()+10
                while time.time()<deadline:
                    if read(root/'agent_state.json')['jobs']['late']['status']=='complete':break
                    time.sleep(.2)
                state=read(root/'agent_state.json')['jobs']['late']
                self.assertEqual(state['status'],'complete');self.assertTrue(state['completion_reconciled']);self.assertEqual(state['attempts'],1)
            finally:p.terminate();p.wait(timeout=10);log.close()

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


class HistoricalConfidenceTests(unittest.TestCase):
    def test_reference_half_width_and_veto_override(self):
        import importlib.util
        import numpy as np
        path=Path(__file__).resolve().parents[1]/'marshawn_actor_confidence.py'
        definition=importlib.util.spec_from_file_location('historical_confidence_test',path)
        module=importlib.util.module_from_spec(definition);definition.loader.exec_module(module)
        actor=np.zeros((220,220),bool);actor[100,100]=True;veto=np.zeros_like(actor)
        weight,_=module.native_confidence(actor,veto,64)
        self.assertAlmostEqual(float(weight[100,132]),.1,places=7)
        self.assertEqual(float(weight[100,164]),0.)
        self.assertEqual(float(weight[100,100]),1.)
        veto[100,100]=True
        weight,_=module.native_confidence(actor,veto,64)
        self.assertEqual(float(weight[100,100]),0.)


class ProductionRecoveryTests(unittest.TestCase):
    def test_orphan_worker_is_rejoined_before_retrying_neural_training(self):
        from production import wait_for_existing_worker
        from agent import completed
        with tempfile.TemporaryDirectory(prefix='marshawn_orphan_worker_') as directory:
            root=Path(directory);marker=root/'locked';receipt=root/'stage_complete.json'
            script='import fcntl,time,json;from pathlib import Path;r=Path('+repr(str(root))+');lease=(r/"worker.lock").open("a+");fcntl.flock(lease,fcntl.LOCK_EX);(r/"locked").write_text("yes");time.sleep(1);(r/"stage_complete.json").write_text(json.dumps(dict(completed_iterations=150000)))'
            child=subprocess.Popen([sys.executable,'-c',script],start_new_session=True)
            try:
                deadline=time.time()+10
                while not marker.exists() and time.time()<deadline:time.sleep(.02)
                self.assertTrue(marker.exists());self.assertFalse(receipt.exists())
                start=time.monotonic();wait_for_existing_worker(root)
                self.assertGreater(time.monotonic()-start,.5)
                self.assertTrue(completed(dict(completion=str(receipt),target=150000,max_optimizer_attempts=150000)))
                child.wait(timeout=5)
            finally:
                if child.poll() is None:child.terminate();child.wait(timeout=5)


class ComparisonTests(unittest.TestCase):
    def test_production_runtime_snapshot_leaves_hpo_paths_identical(self):
        from campaign import runtime_settings
        settings=dict(tools='repo/tools',frozen_helpers='repo/helpers',nerfacc_extension='cache/extension.so',production_tools='campaign/tools',production_frozen_helpers='campaign/helpers',production_nerfacc_extension='campaign/ops/extension.so',production_nerfacc_extension_sha256='expected')
        self.assertEqual(runtime_settings(settings),settings)
        deployed=runtime_settings(settings,production=True)
        self.assertEqual(deployed['tools'],'campaign/tools')
        self.assertEqual(deployed['frozen_helpers'],'campaign/helpers')
        self.assertEqual(deployed['nerfacc_extension'],'campaign/ops/extension.so')
        self.assertEqual(deployed['nerfacc_extension_sha256'],'expected')
        self.assertNotIn('nerfacc_extension_sha256',settings)
        self.assertEqual(settings['tools'],'repo/tools')

    def test_unequal_training_lengths_use_latest_shared_finalist_gate(self):
        from campaign import matched_finalist_candidates
        def row(name,step,value,cap=10):
            return dict(name=name,result=dict(step=step-1,cap=cap,metrics=dict(lpips=value)))
        candidates=[row('early',32000,.18),row('early',64000,.15),row('early',128000,.12),row('late',32000,.17),row('late',64000,.14),row('late',64000,.139,None),row('screen_only',32000,.20)]
        compared,proof=matched_finalist_candidates(candidates,['early','late'])
        self.assertEqual(proof['steps'],64000)
        self.assertEqual({c['result']['step']+1 for c in compared},{64000})
        self.assertEqual(min(compared,key=lambda c:c['result']['metrics']['lpips'])['name'],'late')
        self.assertEqual(len(compared),3)
        with self.assertRaises(ValueError):matched_finalist_candidates(candidates,['missing'])


class SourceCacheTests(unittest.TestCase):
    def test_release_requires_matching_transfer_acknowledgement(self):
        from producer import release
        with tempfile.TemporaryDirectory(prefix='marshawn_source_cache_') as directory:
            root=Path(directory);folder=root/'prepared/000908';folder.mkdir(parents=True)
            atomic(folder/'producer_complete.json',dict(frame=908,native_ready_sha256='expected'))
            (folder/'derived.png').write_bytes(b'cache')
            with self.assertRaises(ValueError):release(root,908,'wrong')
            self.assertTrue((folder/'derived.png').exists())
            release(root,908,'expected');self.assertFalse(folder.exists())
            self.assertTrue(read(root/'prepared_released/000908.json')['released_after_verified_transfer'])
            release(root,908,'expected')


if __name__=='__main__':unittest.main()
