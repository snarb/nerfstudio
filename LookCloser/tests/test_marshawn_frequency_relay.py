"""Frequency publication and provenance safety; never contacts a remote host."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
import torch
SCRIPT=Path(__file__).resolve().parents[1]/'scripts/relay_marshawn_clever_frequencies.py'
loader=importlib.util.spec_from_file_location('frequency_relay_test',SCRIPT)
m=importlib.util.module_from_spec(loader);loader.loader.exec_module(m)


class RelayTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'train_MD.png';source.write_bytes(b'fixture source identity')
        frame=dict(file_path='images/train_MD.png',w=64,h=64)
        metadata=dict(ssim_threshold=.95,train_steps_per_level=1000,eval_patch_batch_size=1024,min_res=16,max_res=8192,n_levels=16,patch_size=8,stride=8,value_type='scalar_resolution',source_image_sha256=m.sha(source),fit_script_sha256=m.FITTER_SHA,image='train_MD.png',image_shape=[64,64],crop_coords_xywh=None)
        tensor=root/'train_MD.pt';sidecar=tensor.with_suffix('.json')
        torch.save(torch.full((8,8),16.),tensor);sidecar.write_text(json.dumps(metadata))
        return source,frame,tensor,sidecar,metadata

    def test_validates_threshold_schedule_source_and_finite_raster(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,frame,tensor,sidecar,metadata=self.fixture(root)
            self.assertEqual(m.validate_pair(tensor,sidecar,frame,source)['shape'],[8,8])
            metadata['ssim_threshold']=.97;sidecar.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):m.validate_pair(tensor,sidecar,frame,source)
            metadata['ssim_threshold']=.95;sidecar.write_text(json.dumps(metadata));torch.save(torch.full((8,8),float('nan')),tensor)
            with self.assertRaises(ValueError):m.validate_pair(tensor,sidecar,frame,source)

    def test_never_overwrites_complete_map_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);incoming=root/'incoming';central=root/'central';incoming.mkdir();central.mkdir()
            self.fixture(incoming)
            (central/'train_MD.pt').write_bytes(b'original complete tensor');(central/'train_MD.json').write_text('{}')
            before=m.sha(central/'train_MD.pt')
            with self.assertRaises(FileExistsError):m.publish_pair(incoming,central,'train_MD',{})
            self.assertEqual(m.sha(central/'train_MD.pt'),before)

    def test_atomic_publication_preserves_partial_and_digests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);incoming=root/'incoming';central=root/'central';incoming.mkdir();central.mkdir()
            self.fixture(incoming);(central/'train_MD.pt').write_bytes(b'crash partial')
            m.publish_pair(incoming,central,'train_MD',dict(camera='MD'))
            self.assertEqual(m.sha(central/'train_MD.pt'),m.sha(incoming/'train_MD.pt'))
            self.assertEqual((incoming/'train_MD.pt.previous_partial').read_bytes(),b'crash partial')
            self.assertTrue((incoming/'publication.json').exists())
            self.assertFalse(list(central.glob('*pending*')))


if __name__=='__main__':unittest.main()
