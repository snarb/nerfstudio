"""Strict dual-source contract for separately released experimental4K production."""
import hashlib
import json
from pathlib import Path
CONTRACT=None

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def verify_files(root,files):
 for name,digest in files.items():
  if Path(name).is_absolute()or'..'in Path(name).parts:raise ValueError('Unsafe inventory path')
  if sha(Path(root)/name)!=digest:raise ValueError('Bound source changed: '+name)

def verify_dataset_layout(data,source_inventory,map_inventory):
 expected=set(source_inventory)|{str(Path(name).relative_to('data'))for name in map_inventory if Path(name).parts[0]=='data'}|{'ready.json'}
 actual={str(p.relative_to(data))for p in Path(data).rglob('*')if p.is_file()}
 if expected!=actual:raise ValueError('Unexpected or missing dataset files: '+repr(sorted(expected^actual)))

def validate_contract(c):
 if c['scope']!='EXPERIMENTAL_4K_ORIGINAL_R0_W64_COLD' or c['start_mode']!='cold' or c['minimum_ancestry_step']!=106316:
  raise ValueError('Exact cold originalR0/W64 and106316 ancestry required')
 if c['bundle']!='4K_RGB_native_confidence_plus_4000_step_maps' or c['pure_resolution_causality']is not False:
  raise ValueError('Explicit resolution-plus-map-budget bundle required')
 if Path(c['data']).resolve()==Path(c['geometry_data']).resolve():raise ValueError('Independent HD geometry source required')
 if c['mapped_support_sha256']!='d60baa9186a63f3769204b93cb46dcfd2ea16e3d41a72ca788934fe19bd2659f':raise ValueError('Original sourcepad1 support required')
 return c

def initialize(contract_path,spec_path,smoke_steps=None):
 global CONTRACT
 if smoke_steps is not None:raise ValueError('Technical smoke cannot enter production output')
 c=validate_contract(read(contract_path));release=read(c['release'])
 if release.get('production_authorized')is not True or release.get('technical_only')is not False:raise ValueError('Explicit production root release required')
 if release.get('contract_sha256')!=sha(contract_path)or release.get('spec_sha256')!=sha(spec_path):raise ValueError('Unbound production release')
 if Path(c['contract_path']).resolve()!=Path(contract_path).resolve()or sha(spec_path)!=c['spec_sha256']:raise ValueError('Wrong contract/spec identity')
 for path,digest in c['artifacts'].items():
  if sha(path)!=digest:raise ValueError('Contract artifact changed: '+path)
 verify_files(c['code_root'],read(c['source_manifest'])['files_sha256'])
 spec=read(spec_path)
 expected=dict(objective='actor',sampling='body80',width=64,radius=0,coarse_step=.00625,distortion_final=.01,seed=42,frequency_threshold=.95,correct_sh_directions=True,density_activation='trunc_exp',frequency_projection_mode='legacy',lr_decay_steps=200000,followup_variant='actor')
 for key,value in expected.items():
  if type(spec.get(key))is not type(value)or spec[key]!=value:raise ValueError('Wrong unchanged recipe '+key)
 if any(k.startswith('technical_')for k in spec)or spec['data']!=c['data']or spec['output']!=c['output']or spec['semantic_admission']['identity_control']is not False:raise ValueError('Unexpected phase/source/output')
 if spec['semantic_admission']['manifest']!=c['semantic_manifest']:raise ValueError('Wrong HD geometry manifest')
 CONTRACT=c

def production_guard(spec_path,smoke_steps):
 if CONTRACT is None or smoke_steps is not None or sha(spec_path)!=CONTRACT['spec_sha256']:raise ValueError('Released production contract required')

def bind_config(cfg):
 if CONTRACT is None:raise ValueError('Production contract not initialized')
 cfg.pipeline.model.semantic_data=CONTRACT['geometry_data']
 return cfg

def validate_data(data):
 from marshawn_semantic_4k_derivation import inspect,validate_map_audit
 c=CONTRACT
 if c is None or Path(data).resolve()!=Path(c['data']).resolve():raise ValueError('Wrong4K source generation')
 ready=read(Path(data)/'ready.json')
 if sha(Path(data)/'ready.json')!=c['ready_sha256']or ready['status']!='EXPERIMENTAL_4K_DATA_ASSEMBLY_READY':raise ValueError('Strict production data-ready receipt required')
 if ready['source_files']!=654 or ready['map_metadata_censor_files']!=189 or ready['bundle']!=c['bundle']or ready['default_promoted']is not False:raise ValueError('Unqualified dataset assembly')
 verify_files(data,read(c['source_inventory']));verify_files(c['candidate_root'],read(c['map_inventory']))
 verify_dataset_layout(Path(data),read(c['source_inventory']),read(c['map_inventory']))
 actual=inspect(c['geometry_data'],data,c['source_inventory'],c['semantic_manifest']);expected=read(c['derivation_receipt'])
 for key in ('source_inventory_sha256','geometry_source_hashes','semantic_manifest_sha256','support','splits','eval9bytes_and_crops_identical','native_RAW_widths_preserved'):
  if actual[key]!=expected[key]:raise ValueError('Dual-source derivation changed: '+key)
 if actual['support']['mapped_bool_sha256']!=c['mapped_support_sha256']:raise ValueError('Semantic support changed')
 validate_map_audit(read(c['aggregate']),actual['splits']['train'])
 technical=read(c['technical_complete']);visual=read(c['technical_visual_review'])
 if technical['status']!='TECHNICAL_PASS_NO_PRODUCTION_RELEASE'or technical['actual_resumed_Adam_updates']!=32 or visual['technical_visual_check_complete']is not True:
  raise ValueError('Technical resume/visual proof missing')
 source=read(Path(data)/'source_metadata.json')
 for key,value in source.items():
  if key in ('training_eligible','frequency_policy'):
   if ready['original_source_generation_status'].get(key)!=value:raise ValueError('Original source status changed')
  elif ready.get(key)!=value:raise ValueError('Ready receipt changed source metadata: '+key)
 if ready['training_eligible']is not True or ready['frequency_policy']!='EXPERIMENTAL_4000_STEP_095_WITH_EXPLICIT_CENSORING':raise ValueError('Explicit assembled map readiness required')
 if ready['source_inventory_sha256']!=sha(c['source_inventory'])or ready['map_inventory_sha256']!=sha(c['map_inventory']):raise ValueError('Ready receipt inventory differs')
 return ready
