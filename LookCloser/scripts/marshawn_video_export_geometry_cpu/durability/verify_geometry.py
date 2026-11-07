"""CPU-only proof for an orphaned render artifact, never a training completion."""
from pathlib import Path
import argparse,json,hashlib

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def owned(root,path):
 p=Path(path).resolve();p.relative_to(Path(root).resolve());return p

def verify(directory,expected):
 out=Path(directory);file=out/'geometry_complete.json'
 if not file.exists():return dict(exists=False,verified=False)
 receipt=read(file);binding=receipt['binding'];record=expected['record'];root=Path(expected['campaign']);frame=root/'frames'/record['frame_id'];data=frame/'data';completion=read(frame/'frame_completion.json');choice=completion['selectors']['actor_lpips'];cp=receipt['checkpoint_binding']
 if receipt['kind']!='NATIVE_MODEL_BOUND_FLOAT_GEOMETRY_V1'or binding['record']!=record or binding['path_sha256']!=expected['path_sha256']or binding['worker_sha256']!=expected['GPU_worker_sha256']:raise ValueError('Wrong render/source/camera artifact')
 if binding['model_frame_id']!=record['frame_id']or binding['frame_completion_sha256']!=sha(frame/'frame_completion.json')or binding['data_ready_sha256']!=sha(data/'ready.json')or binding['transforms_sha256']!=sha(data/'transforms.json'):raise ValueError('Model/source changed after render')
 if binding['code_seal_sha256']!=expected['code_seal_sha256']or sha(root/'code_seal.json')!=expected['code_seal_sha256']:raise ValueError('Wrong sealed model code')
 if completion['status']!='complete'or completion['technical']is not False or not 0<completion['completed_iterations']<=140000 or completion['hard_max_iterations']!=140000:raise ValueError('Wrong training completion family/cap')
 if cp.get('active_hard140k_revision')is not True or cp['checkpoint_sha256']!=choice['checkpoint_sha256']or cp['completed_iterations']!=completion['completed_iterations']or Path(cp['production_campaign'])!=root:raise ValueError('Wrong own selected checkpoint')
 if not receipt['pipeline_tensor_SHA_before']or receipt['pipeline_tensor_SHA_before']!=receipt['pipeline_tensor_SHA_after']:raise ValueError('Renderer changed model buffers')
 parity=receipt['native_HC_parity'];archive=owned(frame/'train/renders',parity['archive_path'])
 if parity['max_RGB_byte_difference']>1 or sha(archive)!=parity['archive_sha256']:raise ValueError('Native HC parity/archive differs')
 tf=read(data/'transforms.json');rows={r['physical_camera']:dict(r,camera_id=i)for i,r in enumerate(tf['frames'])};allowed={r['physical_camera']for r in tf['frames']if r['file_path']in tf['train_filenames']}
 if len(receipt['source_names'])!=16 or set(receipt['source_names'])!=set(receipt['source_rows']):raise ValueError('Wrong donor set')
 for name,source in receipt['source_rows'].items():
  if name not in allowed or source['row']!=rows[name]or source['semantic_sha256']!=sha(data/'semantics'/(name+'.npz'))or source['valid_mask_sha256']!=sha(data/source['row']['mask_path']):raise ValueError('Wrong current donor geometry/semantics')
 for relative,value in receipt['files_sha256'].items():
  path=owned(out,out/relative)
  if Path(relative).is_absolute()or sha(path)!=value:raise ValueError('Float geometry file changed')
 return dict(exists=True,verified=True,kind='HONEST_RENDER_ARTIFACT_ADOPTION_ONLY_NOT_TRAINING_COMPLETION',geometry_receipt_sha256=sha(file),checkpoint_sha256=cp['checkpoint_sha256'],frame=completion['frame'],files_verified=len(receipt['files_sha256']),source_ready_sha256=sha(data/'ready.json'),verifier_sha256=sha(__file__))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--expected-json',required=True);a=p.parse_args();print(json.dumps(verify(a.directory,json.loads(a.expected_json))))
