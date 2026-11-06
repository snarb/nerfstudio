"""Read-only current loss weights on previously qualified TRAIN bag exclusions."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(c,out):
    b=c/'native_semantic_grid_20000/bag_qualification';proof=b/'train_bag_qualification.json'
    if sha(proof)!='5e904fe04a4647a6312496017ae928e24f0077340f93ba016beacebd942d15d7':raise ValueError('Qualified source changed')
    meta=json.loads(proof.read_text())
    if meta['uses_eval_RGB']or meta['uses_eval_metrics']:raise ValueError('Only TRAIN evidence is allowed')
    rows=[]
    for record in meta['records']:
        camera=record['camera'];assert record['split']=='train'
        mask=b/f'{camera}_bag_exclusion_HD.npy';shell=b/f'{camera}_bag_unknown_HD.npy'
        if sha(mask)!=record['exclusion_sha256']or sha(shell)!=record['unknown_sha256']:raise ValueError('Qualified mask hash changed')
        support=np.load(mask,allow_pickle=False);unknown=np.load(shell,allow_pickle=False);path=c/'data/semantics'/f'{camera}.npz'
        with np.load(path,allow_pickle=False)as z:
            if support.dtype!=bool or unknown.dtype!=bool or support.shape!=z['grey'].shape:raise ValueError('Mask geometry differs')
            false=support&z['grey']
            if (support&z['black']).any():raise ValueError('Unexpected black actor exclusion')
            row=dict(camera=camera,qualified_exclusion_pixels=int(support.sum()),original_false_grey_pixels=int(false.sum()),core_sampling_false_grey_pixels=int((false&z['core']).sum()),unknown_shell_pixels=int(unknown.sum()),source_semantic_sha256=sha(path),qualified_exclusion_sha256=sha(mask),unknown_shell_sha256=sha(shell),weights={})
            for name in ['W32','W64']:
                v=z[name][false]
                if not len(v)or not np.isfinite(v).all():raise ValueError('No finite qualified weights')
                row['weights'][name]=dict(positive=int((v>0).sum()),mean=float(v.mean(dtype=np.float64)),min=float(v.min()),max=float(v.max()),sum=float(v.sum(dtype=np.float64)))
        rows.append(row)
    out.mkdir(exist_ok=True,parents=True)
    result=dict(kind='QUALIFIED_STATIONARY_BAG_MAIN_LOSS_READONLY_AUDIT',qualification_sha256=sha(proof),script_sha256=sha(Path(__file__)),qualification_uses_TRAIN_only=True,temporal_proof_paths={n:str(b/n)for n in ['MC_TRAIN_temporal.png','NA_TRAIN_temporal.png']},records=rows,production_masks_modified=False,no_GPU_or_training=True,conclusion='Main loss retains qualified nonactor grey positives in both W32 and W64; fixed semantic support is auxiliary corrected and conservatively treats these interiors/shells as UNKNOWN, not known-free.')
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(output=str(out),false_grey_pixels=sum(r['original_false_grey_pixels']for r in rows),report_sha256=sha(out/'report.json'))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.campaign,a.output)
