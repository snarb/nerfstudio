"""Safety contracts for the cold AABB/halo campaign, without training."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_marshawn_hd_campaign.py'
loader = importlib.util.spec_from_file_location('marshawn_campaign_test', SCRIPT)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)


class CampaignTests(unittest.TestCase):
    def test_aabb_does_not_change_occupancy_opacity_interval(self):
        import sys
        from types import SimpleNamespace as N
        from unittest.mock import patch
        module=N(MarshawnDataParser=object,MarshawnPipeline=object)
        results=[]
        for extent in [.2,3.]:
            model=N(adaptive_coarse_step_size=.001)
            config=N(pipeline=N(model=model,datamanager=N(dataparser=N(),pixel_sampler=N())))
            with patch.dict(sys.modules,{'run_marshawn_hd_campaign':module}):
                m.configure_native(config,dict(aabb=[[-extent]*3,[extent]*3],common_scene_span=7.5,coarse_step=.001,
                    frequency_initialization=dict(policy='zero_explicit')))
            results.append((model.render_step_size,model.occupancy_update_step_size))
        self.assertEqual(results,[(.001,.001),(.001,.001)])

    def test_nested_box_validation(self):
        boxes = {'coordinate_system': 'prepared_world', 'scopes': {
            scope: {size: [[-v]*3, [v]*3] for size, v in [('tight',1),('medium',2),('loose',3)]}
            for scope in ['scene','subject']}}
        m.validate_boxes(boxes)
        boxes['scopes']['scene']['loose'] = [[-1]*3, [1]*3]
        with self.assertRaises(ValueError):
            m.validate_boxes(boxes)

    def test_selector_exact_checkpoint_and_lpips_tie(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [dict(step=4000, eval_all_psnr=30, eval_all_ssim=.9, eval_all_lpips=.2),
                    dict(step=8000, eval_all_psnr=29.93, eval_all_ssim=.9, eval_all_lpips=.1),
                    dict(step=12000, eval_all_psnr=31, eval_all_ssim=.9, eval_all_lpips=.09)]
            # Higher-PSNR unsaved row cannot select a different nearby checkpoint.
            for step in [4000,8000]:
                (root/f'step-{step:09d}.ckpt').touch()
            path = root/'metrics.csv'
            with path.open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader();writer.writerows(rows)
            selected, reason = m.select_checkpoint(path,root)
            self.assertEqual(selected.name, 'step-000008000.ckpt')
            self.assertIn('exact',reason)

    def test_selector_rejects_loss_only_and_out_of_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'step-000022000.ckpt').touch()
            path = root/'metrics.csv'
            path.write_text('step,eval_loss,eval_all_psnr,eval_all_ssim,eval_all_lpips\n4000,.01,,,\n22000,,35,.9,.1\n')
            with self.assertRaises(RuntimeError):
                m.select_checkpoint(path,root)

    def test_selector_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'step-000004000.ckpt').touch()
            path=root/'metrics.csv'
            path.write_text('step,eval_all_psnr,eval_all_ssim,eval_all_lpips\n4000,30,.9,nan\n')
            with self.assertRaises(RuntimeError):m.select_checkpoint(path,root)

    def test_initialization_budget_normalization_and_twelve_arms(self):
        from argparse import Namespace
        from contextlib import redirect_stdout
        import io
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';data.mkdir()
            (data/'transforms.json').write_text('{"orientation_override":"none"}')
            (data/'preparation.json').write_text('{}')
            boxes={'coordinate_system':'prepared_world','normalization_sha256':m.sha(data/'preparation.json'),
                   'scopes':{scope:{size:[[-v]*3,[v]*3] for size,v in [('tight',1),('medium',2),('loose',3)]}
                             for scope in ['scene','subject']}}
            boxfile=root/'boxes.json';boxfile.write_text(json.dumps(boxes))
            args=Namespace(data=data,boxes=boxfile,root=root/'campaign',max_updates=22001,coarse_step=.001,
                           frequency_seed_manifest=None,zero_frequency_init=True)
            with self.assertRaises(ValueError):m.initialize(args)
            args.max_updates=22000
            with redirect_stdout(io.StringIO()):m.initialize(args)
            specs=m.read(args.root/'campaign.json')['variants']
            self.assertEqual(len(specs),12)
            self.assertEqual({m.read(path)['radius'] for path in specs},{0,1})
            self.assertTrue(all(m.read(path)['max_updates']==22000 for path in specs))
            with self.assertRaises(FileExistsError):m.initialize(args)


if __name__=='__main__':unittest.main()
