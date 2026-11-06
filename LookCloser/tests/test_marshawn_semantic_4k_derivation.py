import copy
import importlib.util
from pathlib import Path
import json
from types import SimpleNamespace as N
import pytest
s=importlib.util.spec_from_file_location('derive4k',Path(__file__).parents[1]/'scripts/marshawn_semantic_4k_derivation.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def descriptors():
 names=[f'T{i:02}'for i in range(63)]+['DC','HC','KA'];frames=[]
 for i,camera in enumerate(names):
  frames.append(dict(physical_camera=camera,file_path='images/'+('train_'if i<63 else'eval_')+camera+'.png',mask_path='masks/'+camera+'.png',w=1920,h=1012,fl_x=1355.,fl_y=1353.,cx=960.,cy=506.,transform_matrix=[[1,0,0,i],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
 old=dict(camera_model='OPENCV',frames=frames,train_filenames=[x['file_path']for x in frames[:63]],val_filenames=[x['file_path']for x in frames[63:]],test_filenames=[x['file_path']for x in frames[63:]])
 new=copy.deepcopy(old)
 for r in new['frames'][:63]:
  for k in ('w','h','fl_x','fl_y','cx','cy'):r[k]*=2
 return old,new

def test_split_order_not_global_interleaving():
 a,b=descriptors();evalrow=a['frames'].pop(63);a['frames'].insert(3,evalrow)
 assert m.verify_frames(a,b)['combined_frame_interleaving_may_differ']

@pytest.mark.parametrize('mutation',['pose','intrinsics','eval','membership','order','duplicate'])
def test_bad_derivation_rejected(mutation):
 a,b=descriptors()
 if mutation=='pose':b['frames'][0]['transform_matrix'][0][3]+=.0001
 if mutation=='intrinsics':b['frames'][0]['cx']-=.5
 if mutation=='eval':b['frames'][63]['w']*=2
 if mutation=='membership':b['train_filenames'][0]=b['val_filenames'][0]
 if mutation=='order':b['frames'][0],b['frames'][1]=b['frames'][1],b['frames'][0]
 if mutation=='duplicate':b['frames'][0]['physical_camera']=b['frames'][1]['physical_camera']
 with pytest.raises(ValueError):m.verify_frames(a,b)

def test_missing_or_unbound_maps_cannot_pass_full_gate():
 names=[f'T{i:02}'for i in range(63)];rows=[dict(camera=n,output_hashes={'map':'a','metadata':'b','censor':'c'})for n in names];a=dict(status='EXPERIMENTAL_FREQUENCY_CANDIDATE_COMPLETE',cameras=rows)
 assert m.validate_map_audit(a,names)
 with pytest.raises(ValueError):m.validate_map_audit(dict(a,cameras=rows[:-1]),names)
 rows[0]['output_hashes'].pop('censor')
 with pytest.raises(ValueError):m.validate_map_audit(a,names)

def test_config_adapter_preserves_every_model_choice_and_source_config(tmp_path):
 hd=tmp_path/'hd';high=tmp_path/'high';hd.mkdir();high.mkdir();(hd/'ready.json').write_text('{}');(high/'source').write_text('frozen');inventory=tmp_path/'inventory.json';inventory.write_text(json.dumps({'source':m.sha(high/'source')}));manifest=tmp_path/'manifest.json';manifest.write_text('{}')
 proof=dict(kind='CPU_SOURCE_DERIVATION_ONLY',eval9bytes_and_crops_identical=True,geometry_data=str(hd),training_source_data=str(high),geometry_source_hashes={str(hd/'ready.json'):m.sha(hd/'ready.json')},source_inventory=str(inventory),source_inventory_sha256=m.sha(inventory),semantic_manifest=str(manifest),semantic_manifest_sha256=m.sha(manifest));receipt=tmp_path/'proof.json';receipt.write_text(json.dumps(proof));digest=m.sha(receipt)
 cfg=N(pipeline=N(datamanager=N(data=hd,dataparser=N(data=hd)),model=N(semantic_data=str(hd),density_activation='trunc_exp',radius=0,projection='legacy',semantic_manifest=str(manifest))))
 out=m.bind_config(cfg,receipt,digest)
 assert out.pipeline.datamanager.data==high and out.pipeline.datamanager.dataparser.data==high
 assert vars(out.pipeline.model)==vars(cfg.pipeline.model)
 assert cfg.pipeline.datamanager.data==hd
 (high/'source').write_text('tampered')
 with pytest.raises(ValueError):m.bind_config(cfg,receipt,digest)

def test_inspection_cannot_write_READY_or_source_files(tmp_path):
 for path in [tmp_path/"ready.json",tmp_path/"HD"/"receipt.json",tmp_path/"4K"/"nested"/"receipt.json"]:
  with pytest.raises(ValueError):m.validate_output(path,tmp_path/"HD",tmp_path/"4K")
 m.validate_output(tmp_path/"proposal"/"receipt.json",tmp_path/"HD",tmp_path/"4K")
