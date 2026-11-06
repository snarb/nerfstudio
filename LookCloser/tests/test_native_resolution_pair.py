"""CPU durability/ancestry tests; fake workers never import a model or use a GPU."""
import argparse
import gc
import json
from pathlib import Path
import signal
import socket
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_marshawn_native_resolution_pair as pair
from run_marshawn_native_resolution_chunk import validate_job


def baseline():
    return dict(step=100000, metrics=dict(psnr=30., ssim=.8, lpips=.05, validation_loss=.02))


class PairStateTests(unittest.TestCase):
    def release(self):
        return dict(anchor_step=100000, interval=5000, baseline=baseline(),
                    source=dict(step=100000), arm_order=["HD", "TRAIN4K"])

    def test_anchor_schedule_and_independent_plateaus(self):
        state = pair.initial_state(self.release(), "release")
        self.assertFalse(state["arms"]["HD"]["plateau"]["stopped"])
        for step in [105000, 110000, 115000, 120000]:
            self.assertEqual(pair.next_arm(state, ["HD", "TRAIN4K"]), "HD")
            row = dict(baseline(), step=step)
            state = pair.commit_gate(state, "HD", row, dict(step=step))
            self.assertEqual(pair.next_arm(state, ["HD", "TRAIN4K"]), "TRAIN4K")
            state = pair.commit_gate(state, "TRAIN4K", row, dict(step=step))
        self.assertIsNone(pair.next_arm(state, ["HD", "TRAIN4K"]))
        self.assertEqual(pair.matched_metrics(state)["common_steps"], [100000, 105000, 110000, 115000, 120000])

    def test_baseline_selection_and_invalid_gate(self):
        state = pair.initial_state(self.release(), "release")
        worse = dict(step=105000, metrics=dict(psnr=40., ssim=.9, lpips=.052, validation_loss=.02))
        state = pair.commit_gate(state, "HD", worse, dict(step=105000))
        self.assertEqual(state["arms"]["HD"]["selection"]["step"], 100000)
        self.assertEqual(pair.commit_gate(state, "HD", worse, dict(step=105000)), state)
        with self.assertRaises(ValueError):
            pair.commit_gate(state, "HD", dict(worse, metrics=dict(worse["metrics"], lpips=.04)), dict(step=105000))
        with self.assertRaises(ValueError):
            pair.commit_gate(state, "TRAIN4K", dict(worse, step=110000), dict(step=110000))
        self.assertEqual(pair.matched_metrics(state)["unmatched_steps"]["HD"], [105000])

    def test_release_rejects_unapproved_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "release.json"
            path.write_text(json.dumps(dict(production_launch_authorized=False, restoration_probe_authorized=False)))
            with self.assertRaises(ValueError):
                pair.validate_release(path)
            with self.assertRaises(ValueError):
                pair.validate_release(path, restore_only=True)


class SerialReplayIntegration(unittest.TestCase):
    def test_failed_eval_replays_without_retraining_and_cache_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            worker = root / "worker.py"
            evaluator = root / "evaluator.py"
            worker.write_text('''import argparse,json,hashlib,pathlib
p=argparse.ArgumentParser();p.add_argument('--job');a=p.parse_args();j=json.loads(pathlib.Path(a.job).read_text())
r=json.loads(pathlib.Path(j['root_release']).read_text());root=pathlib.Path(r['output']);live=root/'cache_live'
assert not live.exists();live.write_text(j['arm']);out=pathlib.Path(j['output']);out.mkdir(parents=True)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
with (root/'worker_calls.jsonl').open('a') as f:f.write(json.dumps(dict(arm=j['arm'],source=j['source_step'],target=j['target_step'],restore=j['restore_only']))+'\\n')
ck=out/'checkpoint.ckpt';ck.write_text(json.dumps(dict(step=j['target_step'],arm=j['arm'],source=sha(j['source_checkpoint']))))
rt=out/'runtime.json';rt.write_text('{}');grid=out/'grid.npz';grid.write_bytes(b'grid')
grid.with_suffix('.json').write_text(json.dumps(dict(checkpoint_sha256=sha(ck),grid_archive_sha256=sha(grid))))
n=j['target_step']-j['source_step']
v=dict(kind='CPU_FAKE_ONLY',arm=j['arm'],train_scale=j['train_scale'],restore_only=j['restore_only'],source_step=j['source_step'],final_step=j['target_step'],source_checkpoint_sha256=j['source_checkpoint_sha256'],output_checkpoint=str(ck),output_checkpoint_sha256=sha(ck),output_runtime=str(rt),output_runtime_sha256=sha(rt),output_grid=str(grid),output_grid_sha256=sha(grid),attempts=n,actual_updates=n,initial_adam=j['source_step']-43,final_adam=j['target_step']-43,restored_domain_equality=dict(pipeline=True,optimizers=True,schedulers=True,scalers=True),serialization_roundtrip=dict(pipeline=True,optimizers=True,schedulers=True,scalers=True,rng_state=True),job_sha256=sha(a.job))
(out/'receipt.json').write_text(json.dumps(v));live.unlink()
''')
            evaluator.write_text('''import argparse,json,hashlib,pathlib,sys
p=argparse.ArgumentParser()
for k in ['config','checkpoint','checkpoint-sha256','output','ready','box','v2-manifest','renderer']:p.add_argument('--'+k)
a,extra=p.parse_known_args();out=pathlib.Path(a.output);root=out.parents[2];assert not(root/'cache_live').exists()
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
out.mkdir(parents=True)
if not(root/'failed_once').exists():(root/'failed_once').write_text('preserved failed evaluation');sys.exit(3)
ck=json.loads(pathlib.Path(a.checkpoint).read_text());arm=ck['arm'];m=dict(psnr=30. if arm=='HD' else35.,ssim=.8,lpips=.05 if arm=='HD' else.052,validation_loss=.02)
v=dict(step=ck['step'],checkpoint_sha256=sha(a.checkpoint),ready_sha256=sha(a.ready),box_sha256=sha(a.box),v2_manifest_sha256=sha(a.v2_manifest),no_training_or_model_mutation=True,v1=dict(metrics=m,by_view={k:m for k in ['DC','HC','KA']}))
(out/'evaluation.json').write_text(json.dumps(v))
'''.replace("else35.", "else 35.").replace("else.052", "else .052"))
            source = root / "source.ckpt"
            source.write_text('{"step":100000}')
            runtime = root / "runtime.json"
            runtime.write_text('{}')
            grid = root / "grid.npz"
            grid.write_bytes(b"original_grid")
            grid.with_suffix('.json').write_text(json.dumps(dict(checkpoint_sha256=pair.sha(source), grid_archive_sha256=pair.sha(grid))))
            config = root / "config.yml"
            config.write_text("fake: true\n")
            ready, box, v2 = [root / name for name in ["ready.json", "box.json", "v2.json"]]
            for path in [ready, box, v2]:
                path.write_text('{}')
            cursor = dict(step=100000, checkpoint=str(source), checkpoint_sha256=pair.sha(source),
                          runtime=str(runtime), runtime_sha256=pair.sha(runtime), grid=str(grid), grid_sha256=pair.sha(grid))
            release = dict(production_launch_authorized=True, restoration_probe_authorized=False,
                           anchor_step=100000, interval=5000, arm_order=["HD", "TRAIN4K"],
                           arms={"HD": dict(train_scale=1), "TRAIN4K": dict(train_scale=2)},
                           source=cursor, baseline=baseline(), output=str(output),
                           bound_files={str(path): pair.sha(path) for path in [source, runtime, grid, worker, evaluator, config, ready, box, v2]},
                           python_path=[str(Path(__file__).resolve().parents[1]/"scripts")],
                           hostname=socket.gethostname(), python=sys.executable, GPU_lock=str(root/"gpu.lock"),
                           preserved_HD_history=[cursor], minimum_available_RAM_GiB=20,
                           abort_below_available_RAM_GiB=2, resource_wait_timeout_seconds=1,
                           worker=str(worker), evaluator=str(evaluator), renderer=str(worker),
                           ready=str(ready), box=str(box), v2_manifest=str(v2))
            release_path = root / "release.json"
            release_path.write_text(json.dumps(release))
            original_handlers = {signum: signal.getsignal(signum) for signum in [signal.SIGTERM, signal.SIGINT]}
            for signum, handler in original_handlers.items():
                self.addCleanup(signal.signal, signum, handler)
            real_sleep = time.sleep
            with patch.object(pair, "gpu", return_value={}), patch.object(pair, "memory", return_value=dict(MemAvailable=64*2**30, MemTotal=64*2**30)), patch.object(pair, "materialize_config", return_value=config), patch.object(pair.time, "sleep", side_effect=lambda seconds: real_sleep(.01)):
                with self.assertRaisesRegex(RuntimeError, "Child failed"):
                    pair.main(argparse.Namespace(release=release_path, restore_only=False, resume=False))
                gc.collect()
                state = json.loads((output/"state.json").read_text())
                self.assertEqual(state["pending"]["phase"], "evaluate")
                self.assertEqual(state["pending"]["arm"], "HD")
                self.assertFalse((output/"complete.json").exists())
                pair.main(argparse.Namespace(release=release_path, restore_only=False, resume=True))
            complete = json.loads((output/"complete.json").read_text())
            calls = [json.loads(line) for line in (output/"worker_calls.jsonl").read_text().splitlines()]
            train = [row for row in calls if not row["restore"]]
            self.assertEqual(len(calls), 10)
            self.assertEqual(len(train), 8)
            self.assertEqual([(row["arm"], row["target"]) for row in train], [(arm, step) for step in [105000, 110000, 115000, 120000] for arm in ["HD", "TRAIN4K"]])
            self.assertEqual(complete["matched"]["common_steps"], [100000, 105000, 110000, 115000, 120000])
            self.assertTrue(all(row["stopped"] for row in complete["plateau"].values()))
            self.assertEqual(complete["selections"]["TRAIN4K"]["step"], 100000)
            self.assertEqual(pair.sha(source), cursor["checkpoint_sha256"])
            self.assertFalse((output/"cache_live").exists())
            self.assertNotIn("torch", sys.modules)
            # Pure validation rejects using a technical zero checkpoint as initial ancestry.
            job = pair.make_job(release, release_path, "HD", cursor, config, output/"preflight/test", True)
            job["source_checkpoint_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "Initial ancestry"):
                validate_job(job)


if __name__ == "__main__":
    unittest.main()
