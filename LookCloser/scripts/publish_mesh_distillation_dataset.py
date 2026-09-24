"""Seal an audited dataset and a separately authored visual review/handoff."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess

from joint_temporal_texture import read,sha,atomic_json
from prepare_mesh_distillation_dataset import OUTPUT,verified_copy,verify


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    out=p.parse_args().output;request=verify(out)
    audit=read(out/'independent_audit.json');review=read(out/'visual_review.json')
    assert audit['status']=='pass' and audit['dataset_hashes_sha256']==sha(out/'dataset_hashes.json')
    assert audit['parser_splits']=={'synthetic':{'train':300,'val':24},'real':{'train':62,'val':1}}
    for name,digest in read(out/'dataset_hashes.json').items():
        if sha(out/name)!=digest:raise ValueError('Changed data after audit: '+name)
    assert review['status']=='usable_for_controlled_distillation_with_known_teacher_defects'
    assert review['all_324_views_reviewed_at_contact_sheet_scale']
    assert len(review['contact_sheets'])==14 and review['training_launched'] is False
    status=read(out/'supervisor_status.json')
    assert status['complete']==324 and all(w['exit_code']==0 for w in status['workers'])
    documents=['dec5_mesh_distillation_preparation.md','dec5_mesh_distillation_training_task.md']
    for name in documents:
        verified_copy(Path(__file__).resolve().parents[1]/'experiments'/name,out/name)
    verified_copy(Path(__file__),out/Path(__file__).name)
    retained=['request.json','dataset_hashes.json','audit.json','independent_audit.json',
              'packaging.json','visual_review.json','tests.log','preparation_checks.jsonl',
              'parity/train_0062.json','parity/train_0033.json',*documents,Path(__file__).name]
    retained += ['review/'+n for n in review['contact_sheets']+review['native_panels_reviewed']+['camera_sampling.png']]
    retained += [str(q.relative_to(out)) for q in (out/'heldout_teacher').rglob('*') if q.is_file()]
    atomic_json(out/'complete.json',dict(status='data_preparation_complete_with_disclosed_teacher_defects',
        utc=datetime.now(timezone.utc).isoformat(),frame_id=request['frame'],
        train_views=300,synthetic_validation_views=24,real_train_images=62,real_eval_images=1,
        training_launched=False,frequency_maps_precomputed=False,artifact_free_teacher=False,
        training_integration_pending='mask/confidence objectives, frequency preprocessing and grid-preserving phase transition',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        retained_manifest_and_review_hashes={name:sha(out/name) for name in retained}))
    print('sealed: data preparation complete; no training launched')


if __name__=='__main__':main()
