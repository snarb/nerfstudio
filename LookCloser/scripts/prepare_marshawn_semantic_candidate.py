"""Seal a separate TRAIN-only semantic candidate source/spec; never launch training."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from marshawn_semantic_admission import sha,load_support
from marshawn_semantic_cell_admission import WholeCellAdmission


def main(c,out):
    out.mkdir(parents=True,exist_ok=False);source=out/'source';source.mkdir()
    old=c/'semantic_admission_feasibility_20261106/source'
    for name in ('semantic_grid.npz','grid_receipt.json','support_manifest.json','final_train_ray_receipt.json','witness_receipt.json','partial_scope_receipt.json','unqualified_view_policy.json'):
        shutil.copy2(old/name,source/name)
    shutil.copy2(c/'independent_occupancy/queries/clothing_103_predeclared.json',source/'TRAIN103_queries.json')
    ready=json.loads((c/'data/ready.json').read_text());bounds=np.array(ready['aabb'],dtype=np.float32)
    with np.load(source/'semantic_grid.npz')as z:mapper=WholeCellAdmission(z['semantic_gate_pad1'],z['aabb'])
    mapped=mapper.map_target_grid([128]*3,bounds);cover=WholeCellAdmission(mapped,bounds)
    queries=json.loads((source/'TRAIN103_queries.json').read_text());assert queries['split']=='train'
    for p,r in zip(queries['pixels'],queries['rays']):
        xyz=np.array(p['witness_xyz']);o,d=np.array(r['origin']),np.array(r['direction'])
        if p['camera_name']in ('DC','HC','KA')or not cover.point_admitted(np.array([xyz,o+d*((xyz-o)@d)])).all():raise ValueError('TRAIN source support failure')
    wr=json.loads((source/'witness_receipt.json').read_text());anchors=[p for p in wr['anchors']if p['name']in ('toe_upper','sfm_6862')]
    assert len(anchors)==2 and cover.point_admitted(np.array([a['xyz_floor']for a in anchors])).all()
    meta=dict(kind='reviewed_TRAIN_W64_pad1_whole_LC_cells',train_camera_count=63,files_sha256={p.name:sha(p)for p in sorted(source.iterdir())},
        mapped_bool_sha256=hashlib.sha256(mapped.tobytes()).hexdigest(),all103_witnesses_retained=True,qualified_shoe_anchors_retained=2,
        source_EVAL_masks_used=False,mapping='whole-cell any-intersection, XYZ floor, boundary epsilon2e-7, outside UNKNOWN',native_halo_distinct=True)
    (source/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');digest=sha(source/'manifest.json');load_support(source/'manifest.json',digest,c/'data',bounds,[128]*3)
    spec=dict(data=str(c/'data'),output=str(out/'run'),objective='actor',sampling='body80',width=64,radius=0,coarse_step=.00625,distortion_final=.01,seed=42,frequency_threshold=.95,correct_sh_directions=True,density_activation='trunc_exp',frequency_projection_mode='legacy',lr_decay_steps=200000,followup_variant='actor',semantic_admission=dict(manifest=str(source/'manifest.json'),manifest_sha256=digest,identity_control=False,release_receipt=str(out/'release.json')))
    (out/'spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    for name,identity in [('technical32',False),('identity32',True)]:
        s=dict(spec,output=str(out/name),technical_pause_after=15,semantic_admission=dict(spec['semantic_admission'],identity_control=identity))
        (out/f'{name}.json').write_text(json.dumps(s,indent=2)+'\n')
    print(json.dumps(dict(root=str(out),source_manifest_sha256=digest,training_started=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.campaign,a.output)
