"""Queue contract regression: upstream defaults must not change the fitted SSIM threshold."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/fit_marshawn_frequency_queue.py'


class QueueContractTests(unittest.TestCase):
    def test_explicit_threshold_survives_upstream_default_drift_and_is_receipted(self):
        constructed=[];calls=[]
        class ChangedDefaultConfig:
            def __init__(self,**kwargs):
                constructed.append(kwargs)
                # Deliberately different upstream defaults: the queue must specify
                # its scene contract, rather than inherit a future package change.
                self.ssim_threshold=kwargs.get('ssim_threshold',.999)
                self.train_steps_per_level=kwargs.get('train_steps_per_level',17)
                self.max_res=kwargs.get('max_res',2048)
                self.eval_patch_batch_size=kwargs.get('eval_patch_batch_size',64)
            def _process_single_image(self,**kwargs):
                calls.append(kwargs)
                target=kwargs['save_freq_path']
                target.write_bytes(b'structural fake frequency tensor; no training')
                target.with_suffix('.json').write_text(json.dumps(dict(value_type='scalar_resolution',max_res=self.max_res)))
        package=types.ModuleType('nerfstudio');package.__path__=[]
        scripts=types.ModuleType('nerfstudio.scripts');scripts.__path__=[]
        preprocess=types.ModuleType('nerfstudio.scripts.lookcloser_preprocess')
        preprocess.LookCloserPreprocessConfig=ChangedDefaultConfig
        preprocess.load_image_as_tensor=lambda *args,**kwargs:object()
        torch=types.ModuleType('torch');torch.set_num_threads=lambda value:None
        torch.manual_seed=lambda seed:None;torch.device=lambda name:name
        torch.cuda=types.SimpleNamespace(empty_cache=lambda:None)
        fake_modules={'nerfstudio':package,'nerfstudio.scripts':scripts,
                      'nerfstudio.scripts.lookcloser_preprocess':preprocess,'torch':torch}
        with tempfile.TemporaryDirectory(prefix='queue_contract_no_training_') as tmp:
            root=Path(tmp);(root/'images').mkdir();source=root/'images/train_AA.png'
            source.write_bytes(b'fixture source identity; image decoder is stubbed')
            (root/'transforms.json').write_text(json.dumps(dict(train_filenames=['images/train_AA.png'])))
            with patch.dict(sys.modules,fake_modules),patch.object(sys,'argv',[str(SCRIPT),'--data',str(root)]),patch('os.umask'):
                namespace=runpy.run_path(str(SCRIPT),run_name='queue_contract_test')
                namespace['main']()
            self.assertEqual(len(constructed),1)
            self.assertEqual(constructed[0]['ssim_threshold'],.95)
            self.assertEqual(constructed[0]['train_steps_per_level'],1000)
            self.assertEqual(constructed[0]['max_res'],8192)
            self.assertEqual(constructed[0]['eval_patch_batch_size'],1024)
            self.assertEqual(len(calls),1)
            metadata=json.loads((root/'lookcloser_frequencies/train_AA.json').read_text())
            self.assertEqual(metadata['ssim_threshold'],.95)
            self.assertEqual(metadata['train_steps_per_level'],1000)
            self.assertEqual(metadata['eval_patch_batch_size'],1024)
            self.assertEqual(metadata['source_image_sha256'],hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(metadata['fit_script_sha256'],hashlib.sha256(SCRIPT.read_bytes()).hexdigest())
            self.assertIn('eval_patch_batch_size',metadata)
            self.assertTrue((root/'lookcloser_frequencies/train_AA.pt').is_file())
            self.assertFalse(list((root/'lookcloser_frequencies').glob('*.tmp.pt')))


if __name__=='__main__':unittest.main()
