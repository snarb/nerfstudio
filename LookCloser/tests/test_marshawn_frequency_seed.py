"""Paper seed input corruption, cold/checkpoint behavior and pre-eval checkpoint safety."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/run_marshawn_hd_campaign.py'
loader=importlib.util.spec_from_file_location('seed_controller_test',SCRIPT)
m=importlib.util.module_from_spec(loader);loader.loader.exec_module(m)


def fixture(root):
    data=root/'data';(data/'images').mkdir(parents=True);(data/'lookcloser_frequencies').mkdir()
    image=data/'images/train_AA.png';image.write_bytes(b'fixture source identity')
    frequency=data/'lookcloser_frequencies/train_AA.pt';frequency.write_bytes(b'fixture fitted map identity')
    frequency.with_suffix('.json').write_text('{}')
    (data/'transforms.json').write_text(json.dumps(dict(train_filenames=['images/train_AA.png'],frames=[dict(file_path='images/train_AA.png',physical_camera='AA')])))
    (data/'preparation.json').write_text('{}')
    box=[[-1.,-1.,-1.],[1.,1.,1.]];boxes=root/'boxes.json';boxes.write_text(json.dumps(box))
    seed=root/'seed.npz'
    arrays=dict(indices_xyz=np.array([[1,2,3],[3,4,5]],np.int64),levels=np.array([5,11],np.float32),
        aabb=np.array(box,np.float64),grid_resolution=np.array(128),num_levels=np.array(16),min_res=np.array(16.),max_res=np.array(8192.))
    np.savez(seed,**arrays)
    variant='scene_tight_halo0';manifest=root/'manifest.json'
    payload=dict(kind='frequency_only_train_observed_sfm_seed',uses_eval_rgb=False,occupancy_admission=False,frequency_halo_radius=0,
        transforms_sha256=m.sha(data/'transforms.json'),normalization_sha256=m.sha(data/'preparation.json'),boxes_sha256=m.sha(boxes),train_cameras=1,
        projection_source_sha256=m.sha(SCRIPT.parents[2]/'nerfstudio/utils/lookcloser_projection.py'),
        variants={variant:dict(file=seed.name,sha256=m.sha(seed),aabb=box)},
        map_bindings={'AA':dict(source_sha256=m.sha(image),map_sha256=m.sha(frequency),metadata_sha256=m.sha(frequency.with_suffix('.json')))})
    manifest.write_text(json.dumps(payload))
    policy=m.seed_policy(manifest,variant,box,data,boxes)
    def rewrite(**changes):
        arrays.update(changes);np.savez(seed,**arrays)
        payload['variants'][variant]['sha256']=m.sha(seed);manifest.write_text(json.dumps(payload))
        policy.update(seed_sha256=m.sha(seed),manifest_sha256=m.sha(manifest))
    return policy,arrays,rewrite


def grid(policy):
    from nerfstudio.data.scene_box import SceneBox
    from nerfstudio.model_components.lookcloser_grid import FrequencyGridManager
    return FrequencyGridManager(SceneBox(torch.tensor(policy['aabb'],dtype=torch.float32)),resolution=128,
                                num_levels=16,min_res=16,max_res=8192)


class FrequencySeedTests(unittest.TestCase):
    def test_cold_seed_exact_and_checkpoint_overwrites_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy,arrays,_=fixture(Path(tmp));live=grid(policy)
            receipt=m.initialize_frequency_grid(live,policy)
            indices=torch.from_numpy(arrays['indices_xyz'])
            self.assertTrue(torch.equal(live.grid[indices[:,0],indices[:,1],indices[:,2]],torch.from_numpy(arrays['levels'])))
            self.assertEqual(int(torch.count_nonzero(live.grid)),2)
            self.assertFalse(receipt['occupancy_modified']);self.assertEqual(receipt['frequency_halo_radius'],0)
            live.grid.zero_();live.grid[4,4,4]=7
            checkpoint={k:v.clone() for k,v in live.state_dict().items()}
            restored=grid(policy);m.initialize_frequency_grid(restored,policy);restored.load_state_dict(checkpoint)
            self.assertTrue(torch.equal(restored.grid,live.grid))
            self.assertEqual(float(restored.grid[1,2,3]),0.)

    def test_corrupt_hash_or_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy,_,_=fixture(Path(tmp));Path(policy['seed']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):m.load_frequency_seed(policy)
        with tempfile.TemporaryDirectory() as tmp:
            policy,_,_=fixture(Path(tmp));(Path(policy['data'])/'images/train_AA.png').write_bytes(b'changed')
            with self.assertRaises(ValueError):m.load_frequency_seed(policy)

    def test_valid_hash_cannot_hide_duplicate_nan_bounds_or_unreachable_index(self):
        cases=[dict(indices_xyz=np.array([[1,2,3],[1,2,3]],np.int64)),
               dict(indices_xyz=np.array([[1,2,3],[127,4,5]],np.int64)),
               dict(levels=np.array([5,np.nan],np.float32)),
               dict(levels=np.array([5,16],np.float32)),
               dict(aabb=np.array([[-2.,-1.,-1.],[1.,1.,1.]])),
               dict(grid_resolution=np.array(64))]
        for changes in cases:
            with self.subTest(changes=changes),tempfile.TemporaryDirectory() as tmp:
                policy,_,rewrite=fixture(Path(tmp));live=grid(policy);rewrite(**changes)
                with self.assertRaises(ValueError):m.initialize_frequency_grid(live,policy)
                self.assertEqual(int(torch.count_nonzero(live.grid)),0)

    def test_explicit_zero_and_missing_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy,_,_=fixture(Path(tmp));live=grid(policy)
            with self.assertRaises(ValueError):m.initialize_frequency_grid(live,None)
            self.assertEqual(m.initialize_frequency_grid(live,dict(policy='zero_explicit'))['cells'],0)
            live.grid[0,0,0]=1
            with self.assertRaises(ValueError):m.initialize_frequency_grid(live,dict(policy='zero_explicit'))

    def test_live_aabb_mismatch_is_rejected_before_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy,_,_=fixture(Path(tmp));live=grid(policy);live.aabb_min_buf[0]-=.1
            with self.assertRaises(ValueError):m.initialize_frequency_grid(live,policy)
            self.assertEqual(int(torch.count_nonzero(live.grid)),0)


class CheckpointProtectionTests(unittest.TestCase):
    def trainer(self,step=4000,fail=True):
        class FakeTrainer:
            def __init__(self):
                self.config=SimpleNamespace(steps_per_save=4000,steps_per_eval_batch=4000,steps_per_eval_image=4000,steps_per_eval_all_images=4000)
                self.step=step;self.calls=[];self.optimizer_updates=123
            def save_checkpoint(self,step):self.calls.append(('save',step))
            def eval_iteration(self,step):
                self.calls.append(('eval',step))
                if fail:raise RuntimeError('synthetic evaluator failure')
            def _after_train(self):self.calls.append(('after',self.step))
        m.install_checkpoint_protection(FakeTrainer);m.install_checkpoint_protection(FakeTrainer)
        return FakeTrainer()

    def test_eval_failure_already_has_exact_checkpoint_without_optimizer_update(self):
        trainer=self.trainer()
        with self.assertRaises(RuntimeError):trainer.eval_iteration(4000)
        self.assertEqual(trainer.calls,[('save',4000),('eval',4000)])
        self.assertEqual(trainer.optimizer_updates,123)

    def test_final_failure_preserves_last_update_and_restores_cadence(self):
        trainer=self.trainer(step=21999)
        with self.assertRaises(RuntimeError):trainer._after_train()
        self.assertEqual(trainer.calls,[('save',21999),('eval',21999)])
        self.assertEqual(trainer.config.steps_per_eval_all_images,4000)
        self.assertEqual(trainer.optimizer_updates,123)

    def test_no_unrequested_checkpoint_on_nonboundary(self):
        trainer=self.trainer(fail=False);trainer.eval_iteration(3999)
        self.assertEqual(trainer.calls,[('eval',3999)])


if __name__=='__main__':unittest.main()
