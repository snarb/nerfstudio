"""Structural report tests; fixture values are not scene-quality measurements."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/summarize_marshawn_campaign.py'
loader=importlib.util.spec_from_file_location('marshawn_report_test',SCRIPT)
m=importlib.util.module_from_spec(loader);loader.loader.exec_module(m)


def fixture(root):
    data=root/'data';data.mkdir()
    (data/'transforms.json').write_text(json.dumps(dict(frames=[dict(file_path='images/eval_T.png',physical_camera='T',w=128,h=64)],val_filenames=['images/eval_T.png'])))
    specs=root/'specs';specs.mkdir();paths=[]
    for name,radius in [('scene_tight_halo0',0),('scene_tight_halo1',1)]:
        spec=dict(name=name,radius=radius,scope='scene',size='tight',aabb=[[-1,-2,-3],[1,2,3]],output_dir=str(root/'runs'))
        path=specs/(name+'.json');path.write_text(json.dumps(spec));paths.append(str(path))
    (root/'campaign.json').write_text(json.dumps(dict(data=str(data),variants=paths)))
    directory=root/'runs/scene_tight_halo0/lookcloser/cold_seed42';directory.mkdir(parents=True)
    checkpoint=directory/'step-000004000.ckpt';checkpoint.touch()
    render=directory/'renders';render.mkdir();(render/'eval_img_0000.png').touch()
    payload=dict(selected_checkpoint=str(checkpoint),train_returncode=0,
        eval=dict(checkpoint=str(checkpoint),results=dict(psnr=20.,ssim=.8,lpips=.3,eval_loss=.004),render_dir=str(render)))
    (directory/'run_summary.json').write_text(json.dumps(payload))
    return directory,payload


class ReportTests(unittest.TestCase):
    def test_reports_only_measured_metrics_and_explicit_incomplete(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_report_structure_') as tmp:
            root=Path(tmp);fixture(root)
            report=m.collect(root)
            self.assertEqual(report['branches'][0]['aabb_volume'],48)
            self.assertEqual(report['branches'][0]['metrics'],dict(psnr=20.,ssim=.8,lpips=.3))
            self.assertEqual(report['branches'][0]['status'],'evaluated')
            self.assertEqual(report['branches'][1]['status'],'incomplete')
            self.assertNotIn('eval_loss',json.dumps(report))
            self.assertEqual(report['branches'][0]['checkpoint_id'],'step-000004000.ckpt')

    def test_mismatched_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory,payload=fixture(root)
            payload['eval']['checkpoint']=str(directory/'step-000008000.ckpt')
            (directory/'run_summary.json').write_text(json.dumps(payload))
            self.assertEqual(m.collect(root)['branches'][0]['status'],'invalid_summary')

    def test_nonfinite_results_are_not_reported_as_metrics(self):
        self.assertIsNone(m.finite_metrics(dict(psnr=20.,ssim=.8,lpips=float('nan'))))
        self.assertIsNone(m.finite_metrics(dict(psnr=20.,ssim=.8)))

    def test_report_only_cli_without_image_dependencies(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_report_structure_') as tmp:
            root=Path(tmp);fixture(root);output=root/'report'
            subprocess.run([sys.executable,str(SCRIPT),'--campaign',str(root),'--output',str(output),'--report-only'],check=True,capture_output=True)
            self.assertTrue((output/'summary.json').is_file())
            self.assertTrue((output/'summary.csv').is_file())
            self.assertNotIn('eval_loss',(output/'summary.json').read_text())

    def test_roi_bounds_and_minimum_metric_size(self):
        self.assertEqual(m.validate_roi([0,0,64,64],128,64),[0,0,64,64])
        for box in [[0,0,63,64],[0,0,129,64],[-1,0,100,64]]:
            with self.assertRaises(ValueError):m.validate_roi(box,128,64)

    def test_exact_gt_prediction_panel_extraction_at_smoke_raster(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow is optional for report-only installations')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'eval_img_0000.png'
            source=Image.new('RGB',(256,64),(255,0,0))
            source.paste(Image.new('RGB',(128,64),(0,255,0)),(128,0));source.save(path)
            gt,pred=m.panels(path,128,64)
            self.assertEqual(gt.size,(128,64));self.assertEqual(pred.size,(128,64))
            self.assertEqual(gt.getpixel((127,63)),(255,0,0))
            self.assertEqual(pred.getpixel((0,0)),(0,255,0))
            with self.assertRaises(ValueError):m.panels(path,64,64)


if __name__=='__main__':unittest.main()
